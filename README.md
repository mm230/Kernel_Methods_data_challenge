# DNA Sequence Classification: SARS-CoV-2 (Covid-19) 
The goals of this include studying machine learning techniques, learning how to use
them, and adapting them to structural data. Due of this, we have chosen the challenge of predicting whether a
DNA sequence (or read) belongs to SARS-CoV-2 as our sequence classification task (Covid-19). Here, we outline
the methods we used, our tests, and some of the outcomes. The best submissions were produced by using the
kernel ridge regression (KRR) for the classification on vectorized dataset of the DNA Sequences. </br>


<!-- <br> -->
Refer to this [link](https://www.kaggle.com/competitions/kernel-methods-ammi-2022) to access to the competition 

## Run the project 
To generate a submission file, you can run the `main.py` script by specifying the value of lambda as indicated like the command below:
```
python3 main.py --model kernelridge -l 0.00001
```
if you're using the the Ridge kernel with the ```RBF kernel``` 
Or you can set the value of lambda to 0, if you want to use the Ridge kernel with the polynomial kernel by running the commmand below:
```
python3 main.py --model kernelridgepolynomial -l 0.0

```
</br>

## Decription of the repository
In this repository, there is the `data` folder that contain all the data that we used to achieve the classification task of out project.
The `classifier,py` contain all the classifier used in this project. we used the Kernel Ridge regression.
In the `kernels.py` you will find all the differents kernels that have been implemented (polynomial, Gaussian)

## Results
After trying many different combinations between increasing or decreasing the value of lambda or changing the type of kernel for the classifier, we have obtained some accuracies presented below:

|kernels|Accuracy|
|------|--------|
|rbf kernel|0.97|
|polynomial kernel|0.93|

## Requirement installation
To run this, make sure to install all the requirements by:

```
$ pip install -r requirements.txt 
```


# Contributors 
<div style="display:flex;align-items:center">

<div style="display:flex;align-items:center">
    <div>
        <h5> <a href='..'> Mouhamadou Mansour Sow (Barro) </a> </h5> <img src="data/barro.jpeg" height= 7% width= 7%>
<div>
    <h5> <a href='.'> Albert Agisha </a> </h5> <img src="data/albert.png" height= 7% width= 7%>
    
<div>
</div>