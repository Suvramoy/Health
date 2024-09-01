import pickle

model = pickle.load(open(r'models\Heart_pipeline_gb.pkl', 'rb'))
print(type(model))