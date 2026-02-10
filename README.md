

# Deep-Schrodinger

Este projeto implementa e demonstra a aplicação do método de Galerkin Profundo (Deep Galerkin Method) para a solução da equação de Schrödinger estacionária. A teoria física e matemática aplicada ao código e experimentos apresentados aqui são resultados que começaram no Trabalho de Conclusão de Curso (TCC) "APLICAÇÃO DO MÉTODO DE GALERKIN PROFUNDO NA SOLUÇÃO DA EQUAÇÃO DE SCHRÖDINGER ESTACIONÁRIA UNIDIMENSIONAL", disponível na biblioteca da Universidade Federal do Cariri (UFCA).


## Objetivo do Projeto

Resolver numericamente a equação de Schrödinger para potenciais reais utilizando o método redes neurais profundas de Galerkin profundo, comparando os resultados com métodos tradicionais (como diferenças finitas) e visualizando as funções de onda aprendidas.


## Estrutura do Projeto

De modo geral, os arquivos estão separados por dimensão, método de implementação do MGP (método de Galerkin Profundo) e potencial utilizado. Sendo o metodo_01 o primeiro método que foi  implementado.

Temos duas pastas principais:

- **src/**: Implementação dos métodos numéricos, funções de perda, arquitetura da rede neural (apenas DGMNet1D no momento) e utilitários de visualização.
- **examples/**: Exemplos de uso.

Em cada pasta de dimensão e potencial correspondente em **src/** temos:

- **metodos_numericos/**: métodos numéricos tradicionais (como o método das diferenças finitas) implementados para comparação de resultados e métricas com o MGP em questão.

- **modelos/**: Pesos de modelos já treinados para diferentes estados quânticos.



## Exemplo de uso discutido no trabalho motivador do projeto

1. Instale as dependências necessárias (TensorFlow, NumPy, Matplotlib).
2. Execute o arquivo de exemplo em `examples/1D/metodo_01/morse_1d_h2/example.py` para:
   - Carregar e visualizar modelos já treinados.
   - Treinar um novo modelo do zero.
   - Visualizar as funções de onda aprendidas.
   - Salvar checkpoints e modelos.

O arquivo de exemplo realiza as seguintes etapas:
- Carrega um modelo treinado para o estado fundamental do potencial de Morse.
- Plota a função de onda aprendida.
- Treina um novo modelo para outros estados, utilizando diferentes funções de perda físicas.
- Salva e avalia o progresso do treinamento, incluindo checkpoints.

Os resultados obtidos a partir deste código de exemplo foram validados e apresentados no TCC, mostrando excelente concordância com métodos tradicionais e destacando o potencial das redes neurais profundas para problemas de física quântica, motivando o projeto.


## Objetivo do Repositório

Compilar diferentes implementações do DGM para resolver equação de Schrödinger estacionária, na maior variabilidade de problemas, potenciais e dimensões espaciais possíveis. Por isso, os arquivos estão organizados por método de implementação (ainda há possibilidades arquitetônicas para MGP), potencial/problema físico e dimensão espacial.

## Autor

**Igor Soares**

- 📧 E-mail: igorsoarescontaoo@gmail.com  
- 💻 GitHub: [@igor439](https://github.com/igor439)
