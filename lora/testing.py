import joblib
import dill
import numpy as np
from keras.models import load_model

model = joblib.load(r"random_forest.joblib")


def model_imp(distance, m, var):
    #    from keras.models import load_model

    d = distance
    power_array = [x for x in range(5, 21, 3)]
    Output = np.zeros((6, 2))
    Output2 = np.zeros((6, 2))
    # model = load_model('PowerOptimizer.h5')

    if var == 'tf':
        model = load_model(r'D:\Iot Research\Code\IoT-MAB2 (windows)\IoT-MAB2\lora\PowerOptimizer.h5')
        # else:
        model2 = joblib.load(r"random_forest2.joblib")

    if var == 'tf':

        for SF in range(7, 13):
            for CR in range(5, 8, 2):
                inpt = np.array([[SF / 12], [CR / 7], [distance / 4500], [m / 100]])
                inpt = inpt.reshape(1, -1)
                Power = model(inpt)
                index = np.argmax(Power[0], axis=0)

                if index == 6:
                    Output[int(SF - 7)][int((CR - 5) / 2)] = 23  # None For now
                else:
                    # print('he')
                    # print(SF - 7)
                    # print((CR - 5) / 2)
                    Output[int(SF - 7)][int((CR - 5) / 2)] = power_array[index]
    # else:
    for SF in range(7, 13):
        for CR in range(5, 8, 2):
            inpt = np.array([[distance], [SF], [CR], [m]])
            inpt = inpt.reshape(1, -1)
            # print(inpt)
            Power = model2.predict(inpt)
            if Power == 23:
                Output2[int(SF - 7)][int((CR - 5) / 2)] = 23  # None For now
            else:
                # print('he')
                # print(SF - 7)
                # print((CR - 5) / 2)
                Output2[int(SF - 7)][int((CR - 5) / 2)] = Power  # power_array[index]
    print(f"{Output} and {Output2} with {Output - Output2}")
    # return Output


model_imp(3000, 100, 'tf')

model_imp(2000, 100, 'tf')

model_imp(4000, 1, 'tf')

model_imp(3500, 1, 'tf')

model_imp(4500, 1, 'tf')
