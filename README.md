END to END artificial neural network model for the binary prediction of molecules as EGFR inhibitors using python script. 

*****

After installing the requirements in an Virtual Environment
or creating a new environment using conda by using "tfgpu.yml" file
just open the command prompt or Terminal in the above created virtual environment
run the command given below
### Using Tensorflow to make prediction
python predict_TF_rdkit.py test.smi


similarly, for any unknown molecule when just use the command by specifying the path
of "*.smi" file.
The supplied "test.smi" file is an Inactive molecule.

python predict_name_of_algorithm.py [specify_path]*.smi


The result will be displayed in the terminal as Molecule to be active or Inactive

The image in repository indicate the training and validation accuracy of the Tensorflow Model.
For the trained model after Hyperparameter tuning,We achieved prediction stats were as mentioned below:
Average Training Accuracy: 0.9487421000003815
Average Validation Accuracy: 0.8665430080890656





 
