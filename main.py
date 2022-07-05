import argparse
from kernels import *
from classifiers import *
import pandas as pd


X_train = pd.read_csv('./data/Xtr_vectors.csv')
X_test = pd.read_csv('./data/Xte_vectors.csv')
Y_train = pd.read_csv('./data/Ytr.csv')

x_test1 = X_test.copy()


parser = argparse.ArgumentParser()

parser.add_argument("-m" ,"--model_name", help = "This is the name of the model", required = True)
parser.add_argument("-l" ,"--lambda", help = "The value of lambda",type = float, required = True)
parser.add_argument("-s" ,"--sigma", help = "Valkue of sigma",required = False)

mains_args = vars(parser.parse_args())

lambd=mains_args["lambda"]
sigma=mains_args["sigma"]


X_train_vec = X_train.drop("Id", axis = 1).values
X_test_vec = X_test.drop("Id", axis = 1).values
Y_train_vec = Y_train.drop("Id", axis = 1).values

model = KernelRidge(lambd=lambd,sigma=sigma)
print("Avant prediction")
ypred = model.fit(X_train_vec, Y_train_vec).predict(X_test_vec)
print("Apres prediction",ypred)


# model = KernelRidge(lambd=0.1, sigma = 0.01)
# y_pred = model.fit(X_train_vec, Y_train_vec)
# prediction = model.predict(X_test_vec)

# model = KernelRidge_polynomial()
# y_pred_polynomial = model.fit(X_train_vec, Y_train_vec)
# prediction_polynomial = model.predict(X_test_vec)

if mains_args["model_name"].lower() == "kernelridge":
    model = KernelRidge(lambd=lambd, sigma = sigma)
    y_pred = model.fit(X_train_vec, Y_train_vec)
    prediction = model.predict(X_test_vec)
    y_pred = np.array([ 0 if p < 0.5 else 1 for p in prediction]).reshape(len(prediction),1)
    iD = pd.DataFrame(x_test1.Id)
    pred = pd.DataFrame(y_pred, columns = ["Covid"])
    results = pd.DataFrame(np.hstack([iD, pred]), columns = ["Id", "Covid"])
    results.to_csv("./submissions/prediction_for_Ridge_kernel.csv",index=False)
elif mains_args["model_name"].lower() == "kernelridgepolynomial":
    # instantiate kernel Ridge classifier using polynomial kernel model
    model = KernelRidge_polynomial()
    y_pred_polynomial = model.fit(X_train_vec, Y_train_vec)
    prediction_polynomial = model.predict(X_test_vec)
    y_pred_polynomial = np.array([ 0 if p < 0.5 else 1 for p in prediction_polynomial]).reshape(len(prediction_polynomial),1)
    iD_poly = pd.DataFrame(x_test1.Id)
    pred_poly = pd.DataFrame(y_pred_polynomial, columns = ["Covid"])

    results_poly = pd.DataFrame(np.hstack([iD_poly, pred_poly]), columns = ["Id", "Covid"])
    results_poly.to_csv("./submissions/prediction_for_Ridge_kernel_polynomial.csv",index=False)

print("✅✅✅✅✅✅✅")


