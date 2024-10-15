# define number of classes based on YAML
import yaml

# Importação  modelo yolo previamente definido no arquivo index.py
from index import model

# data.yaml contém informações sobre o conjunto de dados  e nele estão as classes presentes nas imagens
#  aberto em modo de leitura ('r').
# Carrega o conteúdo do arquivo YAML e extrai o número de classes ('nc')
# O número de classes é convertido em uma string e armazenado na variável num_classes.
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# diretório de destino para resultados do treinamento
project = "/home/naruto/PycharmProjects/pythonProject/v3"

# Define um subdiretório específico para este treinamento
name = "300_epochs-"


# treinamento do modelo

# método train do objeto model
results = model.train(data='/home/naruto/PycharmProjects/pythonProject/v2/data.yaml',
                      project=project,
                      name=name, # subdiretório
                      epochs=300,
                      patience=20, #N epochs para esperar sem melhoria nas métricas de validação antes de parar o treinamento antecipadamente
                      batch=16, #
                      imgsz=640,#

)

