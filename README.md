# Text Classification
Classifying text content for Vietnamese

# Resources
Dataset: <br> 
[Google search results](https://www.google.com/search?q=A+Large-scale+Vietnamese+News+Text+Classification+Corpus&ei=T_3DYtH2NPKU3LUPk6qC4As&ved=0ahUKEwiR2u__sOH4AhVyCrcAHROVALwQ4dUDCA4&uact=5&oq=A+Large-scale+Vietnamese+News+Text+Classification+Corpus&gs_lcp=Cgdnd3Mtd2l6EANKBAhBGABKBAhGGABQAFgAYNAFaABwAXgAgAGCAogBggKSAQMyLTGYAQCgAQKgAQHAAQE&sclient=gws-wiz) <br>
Word vectors:<br> 
https://github.com/Kyubyong/wordvectors

# Tasks
* Cleaning the text, splitting it into words and handling punctuation and case.
* Categorizing text data.
* Building the models.
* Model evaluation.
* Building RESTful API
* Building web/app layout.

# Work flow
* Install the packages: `pip install -r setup.txt`
* Download the dataset and extract into `./data`
---
* Run `python build.py` or `build.py` to build data
---
* Run `python train.py` or `train.py` for full training data
---
* Run `python predict.py` or `predict.py` to predict the result
    <br>_You can also change the algorithm in this._