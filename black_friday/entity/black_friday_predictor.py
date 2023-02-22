import os, sys
from black_friday.exception import BlackFridayException
from black_friday.util.util import load_object
import pandas as pd

class BlackFridayData:
    def __init__(self,
                 Gender : object,
                 Age : object,
                 Occupation : int,
                 City_Category : object,
                 Stay_In_Current_City_Years : object,
                 Marital_Status : int,
                 Product_Category_1 : int,
                 Product_Category_2 : int,
                 Product_Category_3 : int):
        try:
            self.Gender = Gender
            self.Age = Age
            self.Occupation = Occupation
            self.City_Category = City_Category
            self.Stay_In_Current_City_Years = Stay_In_Current_City_Years
            self.Marital_Status = Marital_Status
            self.Product_Category_1 = Product_Category_1
            self.Product_Category_2 = Product_Category_2
            self.Product_Category_3 = Product_Category_3
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def black_friday_data_frame(self):
        try:
            data = self.get_black_friday_data_as_dict()
            df = pd.DataFrame(data=data)
            return df
        except Exception as e:
            raise BlackFridayException(e, sys) from e 
        
    def get_black_friday_data_as_dict(self):
        try:
            input_data = {
                'Gender' : [self.Gender],
                'Age' : [self.Age],
                'Occupation' : [self.Occupation],
                'City_Category' : [self.City_Category],
                'Stay_In_Current_City_Years' : [self.Stay_In_Current_City_Years],
                'Marital_Status' : [self.Marital_Status],
                'Product_Category_1' : [self.Product_Category_1],
                'Product_Category_2' : [self.Product_Category_2],
                'Product_Category_3' : [self.Product_Category_3]
            }
            return input_data
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    
class BlackFridayPredictor:
    def __init__(self, model_dir):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_latest_model(self):
        try:
            folder_list = list(map(int, os.listdir(self.model_dir)))
            folder_name = os.path.join(self.model_dir, f"{max(folder_list)}")
            file_name = os.listdir(folder_name)[0]
            latest_model_file_path = os.path.join(folder_name, file_name)
            return latest_model_file_path
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def predict(self, X):
        try:
            model_file_path = self.get_latest_model()
            
            model = load_object(model_file_path)
            
            columns=['Gender', 'Age', 'Occupation', 'City_Category', 
                     'Stay_In_Current_City_Years', 'Marital_Status',
                     'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
            
            X = X[columns]
            
            Purcahse = model.predict(X)
            
            return Purcahse
        
        except Exception as e:
            raise BlackFridayException(e, sys) from e