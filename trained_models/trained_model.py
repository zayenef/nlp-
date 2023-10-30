from keras.models import load_model
import pickle

####################models1###########
model_paths1 = [
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models1\\cAGR_model1.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models1\\cCON_model1.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models1\\cEXT_model1.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models1\\cOPN_model1.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models1\\cNEU_model1.h5'
]
models1 = {}
for model_path in model_paths1:
    trait = model_path.split('\\')[-1].split('_')[0]
    model = load_model(model_path)
    models1[trait] = model

#print("Contents of models1:")
#print(models1)
###################models2############

model_paths2 = [
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models2\\cAGR_model2.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models2\\cCON_model2.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models2\\cEXT_model2.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models2\\cOPN_model2.h5',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_models2\\cNEU_model2.h5'
]
models2 = {}
for model_path in model_paths2:
    trait = model_path.split('\\')[-1].split('_')[0]
    model = load_model(model_path)
    models2[trait] = model

#print("Contents of models2:")
#print(models2)
##############meta_models#############

model_paths3 = [
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_meta_models\\cAGR_meta_models_model.pkl',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_meta_models\\cCON_meta_models_model.pkl',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_meta_models\\cEXT_meta_models_model.pkl',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_meta_models\\cNEU_meta_models_model.pkl',
    'C:\\Users\\DELL\\OneDrive\\Desktop\\nlp models\\best_meta_models\\cOPN_meta_models_model.pkl'
]
meta_models = {}
for model_path in model_paths3:
    trait = model_path.split('\\')[-1].split('_')[0]
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    meta_models[trait] = model

#print("Contents of meta_models:")
#print(meta_models)