import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.stdout.reconfigure(encoding='utf-8')

# Ensure we import metrics from local utilities.py
sys.path.append('.')
from utilities import (
    DataGenerator, focal_tversky, tversky_loss, tversky,
    dice_coefficient, dice_loss, bce_dice_loss,
    iou_score, sensitivity, specificity, precision_metric
)

# ============== 162-LAYER ATTENTION RESUNET ARCHITECTURE ==============

def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape([1, 1, filters])(se)
    return Multiply()([input_tensor, se])

def resblock(X, f, use_se=True):
    X_copy = X
    # Main path
    X = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), 
               kernel_initializer='he_normal', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f, kernel_size=(3, 3), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    # Short path
    X_copy = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), 
                    kernel_initializer='he_normal', padding='same')(X_copy)
    X_copy = BatchNormalization()(X_copy)
    # Add paths
    X = Add()([X, X_copy])
    X = Activation('relu')(X)
    if use_se:
        X = squeeze_excite_block(X)
    return X

def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    phi_g = Conv2D(inter_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(g)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(f)
    rate = Activation('sigmoid')(psi_f)
    return Multiply()([x, rate])

def upsample_concat(x, skip, use_attention=True):
    x = UpSampling2D((2, 2))(x)
    if use_attention:
        skip = attention_gate(skip, x, skip.shape[-1] // 2)
    merge = Concatenate()([x, skip])
    return merge

def build_attention_resunet(input_shape=(256, 256, 3)):
    X_input = Input(input_shape)
    
    # ============== ENCODER ==============
    conv1_in = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(X_input)
    conv1_in = BatchNormalization()(conv1_in)
    conv1_in = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_in)
    conv1_in = BatchNormalization()(conv1_in)
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv1_in)
    
    conv2_in = resblock(pool_1, 64)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv2_in)
    
    conv3_in = resblock(pool_2, 128)
    pool_3 = MaxPool2D(pool_size=(2, 2))(conv3_in)
    
    conv4_in = resblock(pool_3, 256)
    pool_4 = MaxPool2D(pool_size=(2, 2))(conv4_in)
    
    # ============== BOTTLENECK ==============
    conv5_in = resblock(pool_4, 512)
    conv5_in = Conv2D(512, 3, padding='same', dilation_rate=2, kernel_initializer='he_normal')(conv5_in)
    conv5_in = BatchNormalization()(conv5_in)
    conv5_in = Activation('relu')(conv5_in)
    
    # ============== DECODER ==============
    up_1 = upsample_concat(conv5_in, conv4_in, use_attention=True)
    up_1 = resblock(up_1, 256)
    
    up_2 = upsample_concat(up_1, conv3_in, use_attention=True)
    up_2 = resblock(up_2, 128)
    
    up_3 = upsample_concat(up_2, conv2_in, use_attention=True)
    up_3 = resblock(up_3, 64)
    
    up_4 = upsample_concat(up_3, conv1_in, use_attention=True)
    up_4 = resblock(up_4, 32)
    
    # ============== OUTPUT ==============
    output = Dropout(0.3)(up_4)
    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(output)
    
    return Model(inputs=X_input, outputs=output)

# ============== 90-LAYER SIMPLE RESUNET ARCHITECTURE ==============

def build_simple_resunet(input_shape=(256, 256, 3)):
    def simple_resblock(X, f):
        X_copy = X
        X = Conv2D(f, kernel_size=(1,1), strides=(1,1), kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(f, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X_copy = Conv2D(f, kernel_size=(1,1), strides=(1,1), kernel_initializer='he_normal')(X_copy)
        X_copy = BatchNormalization()(X_copy)
        X = Add()([X, X_copy])
        X = Activation('relu')(X)
        return X
    
    def simple_upsample_concat(x, skip):
        x = UpSampling2D((2,2))(x)
        merge = Concatenate()([x, skip])
        return merge
    
    X_input = Input(input_shape)
    
    # Encoder
    conv1_in = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(X_input)
    conv1_in = BatchNormalization()(conv1_in)
    conv1_in = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_in)
    conv1_in = BatchNormalization()(conv1_in)
    pool_1 = MaxPool2D(pool_size=(2,2))(conv1_in)
    
    conv2_in = simple_resblock(pool_1, 32)
    pool_2 = MaxPool2D(pool_size=(2,2))(conv2_in)
    
    conv3_in = simple_resblock(pool_2, 64)
    pool_3 = MaxPool2D(pool_size=(2,2))(conv3_in)
    
    conv4_in = simple_resblock(pool_3, 128)
    pool_4 = MaxPool2D(pool_size=(2,2))(conv4_in)
    
    conv5_in = simple_resblock(pool_4, 256)
    
    # Decoder
    up_1 = simple_upsample_concat(conv5_in, conv4_in)
    up_1 = simple_resblock(up_1, 128)
    
    up_2 = simple_upsample_concat(up_1, conv3_in)
    up_2 = simple_resblock(up_2, 64)
    
    up_3 = simple_upsample_concat(up_2, conv2_in)
    up_3 = simple_resblock(up_3, 32)
    
    up_4 = simple_upsample_concat(up_3, conv1_in)
    up_4 = simple_resblock(up_4, 16)
    
    output = Conv2D(1, (1,1), padding="same", activation="sigmoid")(up_4)
    
    return Model(inputs=X_input, outputs=output)

# ============== DATA PREPARATION & TRAINING ==============

def main():
    print("📂 Loading data...")
    csv_path = 'data_mask.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Keep only positive scans (mask == 1) for segmentation task
    df_mask = df[df['mask'] == 1].copy()
    print(f"📊 Total samples with tumor: {len(df_mask)}")
    
    # Train/Val/Test Split (85% train/val, 15% test)
    X_train, X_val = train_test_split(df_mask, test_size=0.15, random_state=42)
    X_test, X_val = train_test_split(X_val, test_size=0.5, random_state=42)
    
    print(f"   - Train samples: {len(X_train)}")
    print(f"   - Val samples: {len(X_val)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Create generators
    train_gen = DataGenerator(
        ids=list(X_train.image_path),
        mask=list(X_train.mask_path),
        batch_size=16,
        shuffle=True
    )
    
    val_gen = DataGenerator(
        ids=list(X_val.image_path),
        mask=list(X_val.mask_path),
        batch_size=16,
        shuffle=False
    )
    
    test_gen = DataGenerator(
        ids=list(X_test.image_path),
        mask=list(X_test.mask_path),
        batch_size=16,
        shuffle=False
    )
    
    # ---------------- 1. TRAIN ATTENTION RESUNET ----------------
    print("\n🏗️ Building Attention ResUNet (162 layers)...")
    model_att = build_attention_resunet()
    
    # Save model structure as JSON
    print("💾 Saving Attention ResUNet JSON configuration...")
    with open('ResUNet-model.json', 'w') as json_file:
        json_file.write(model_att.to_json())
        
    model_att.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_tversky,
        metrics=[tversky, dice_coefficient, iou_score, sensitivity, specificity, precision_metric]
    )
    
    print("🚀 Training Attention ResUNet...")
    callbacks_att = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath="ResUNet-weights.keras", monitor='val_dice_coefficient', mode='max', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]
    
    history_att = model_att.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks_att,
        verbose=1
    )
    
    # ---------------- 2. TRAIN SIMPLE RESUNET ----------------
    print("\n🏗️ Building Simple ResUNet (90 layers)...")
    model_simple = build_simple_resunet()
    
    # Save model structure as JSON
    print("💾 Saving Simple ResUNet JSON configuration...")
    with open('ResUNet-MRI.json', 'w') as json_file:
        json_file.write(model_simple.to_json())
        
    model_simple.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_tversky,
        metrics=[tversky, dice_coefficient, iou_score, sensitivity, specificity, precision_metric]
    )
    
    print("🚀 Training Simple ResUNet...")
    callbacks_simple = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath="weights_seg.hdf5", monitor='val_dice_coefficient', mode='max', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]
    
    history_simple = model_simple.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks_simple,
        verbose=1
    )
    
    # ---------------- EVALUATION ----------------
    print("\n📊 Evaluating models on Test Set...")
    
    eval_att = model_att.evaluate(test_gen, verbose=1)
    print("\n========================================")
    print("Attention ResUNet Test Metrics:")
    for name, val in zip(model_att.metrics_names, eval_att):
        print(f"  - {name}: {val:.4f}")
        
    eval_simple = model_simple.evaluate(test_gen, verbose=1)
    print("\n========================================")
    print("Simple ResUNet Test Metrics:")
    for name, val in zip(model_simple.metrics_names, eval_simple):
        print(f"  - {name}: {val:.4f}")
    print("========================================\n")
    
    print("✓ Model retraining and evaluation complete!")

if __name__ == '__main__':
    main()
