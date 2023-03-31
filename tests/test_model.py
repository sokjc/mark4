import pytest 

from mark4.model import MyModel

class TestModel:

    def test_build_model(self):

        model = MyModel(
                num_layers=3, 
                num_nodes=5, 
                num_classes=2)
        
        assert model.get_config()['num_classes'] == 2
