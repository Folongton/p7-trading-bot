import os
from src.common.globals import G
PROJECT_PATH = G.get_project_root()

def check_model_presense(model_name):
    ''' Checks if model is already in the folder "models_trained" recursively.
        If so, returns True, else False.

    INPUT:  model_name: str
    OUTPUT: bool
    '''
    print (os.path.join(str(PROJECT_PATH), r'models_trained'))
    model_present = False
    scalers_present = False
    for dirpath, dirnames, filenames in os.walk(os.path.join(str(PROJECT_PATH), r'models_trained')):
        
        for filename in filenames:
            print (filename)
            if model_name in filename and filename.endswith('.keras'):
                model_present = True

            if model_name in filename and filename.endswith('_scalers.pkl'):
                scalers_present = True
            print (model_present, scalers_present)


    if model_present and scalers_present:
        return True
    elif model_present and not scalers_present:
        raise Exception(f'Model {model_name} is present, but scalers are not')
    elif not model_present and scalers_present:
        raise Exception(f'Model {model_name} is not present, but scalers are')
    else:
        return False
        
check_model_presense('MSFT_LSTM_W20_SBS5500_B32_E500_P42113_2023_10_09__16_15')