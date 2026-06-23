def classify_severity(tumor_percentage):
    if(tumor_percentage < 2):
        return "Small"
    elif(tumor_percentage < 8):
        return "Medium"
    elif(tumor_percentage<15):
        return "Large"
    return "Critical"
