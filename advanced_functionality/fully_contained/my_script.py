import pickle

model = pickle.load(open('models/model.pkl', 'rb'))
transformer = pickle.load(open('models/transformer.pkl', 'rb'))