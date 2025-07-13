import torch

TARGET_COLUMN_NAMES = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
TARGET_COLUMNS_MAPPING = {
    'X4_mean': 'X4', 
    'X11_mean': 'X11', 
    'X18_mean': 'X18', 
    'X26_mean': 'X26', 
    'X50_mean': 'X50', 
    'X3112_mean': 'X3112',
}
SUBMISSION_COLUMNS = ['id', 'X4', 'X11', 'X18', 'X50', 'X26', 'X3112']
STD_COLUMN_NAMES = ['X4_sd', 'X11_sd', 'X18_sd', 'X26_sd', 'X50_sd', 'X3112_sd']
DTYPE = torch.float32
