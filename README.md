Para executar o código, duas biblotecas do Python são necessárias:
 - numpy
 - matplotlib


Pode-se executar o código de duas forma:
1) Pelo notebook (arquivo .ipynb). Pode-se abrir o arquivo no Jupyter e executá-lo normalmente, fixando os valores dos parâmetros manualmente.

2) rodando o comando 
	"python tp1-matheus_barros_pereira.py [dataset] [pop_size] [ngenerations] [tournament] [crossover_prob] [mutation_prob] [elitism]"
onde dataset é o nome da base que se encontra dentro da pasta "datasets"; pop_size é um inteiro que representa o tamanho da população; ngenerations é um inteiro que representa o número de geraçes; tournament é um inteiro que representa o tamanho do torneio; crossover_prob é um float que representa a probabilidade de cruzamento; mutation_prob é um float que representa a probabilidade de mutação e elitism é uma string que pode ser "True" ou "False", representado o uso ou não dos operadores elitistas.

Por exemplo, para rodar o código no dataset "synth1", com população de tamanho 50, 50 gerações, torneio de tamanho 2, 0.9 de taxa de cruzamente, 0.05 taxa de mutação e com operadores elististas, o comando seria:
python tp1-matheus_barros_pereira.py synth1 50 50 2 0.9 0.05 True

Os gráficos serão salvos na pasta "graphics"
