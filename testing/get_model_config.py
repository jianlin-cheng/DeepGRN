from keras.models import model_from_json
with open ("model.json", "r") as myfile:
    json_string = myfile.readlines()[0]
model = model_from_json(json_string)
