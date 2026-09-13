# entrega_tc_3_grupo_15

## Decisões arquiteturais
### Processamento Batch

Decidimos utilizar processamento Batch devido à simplicidade da arquitetura, à flexibilidade na execução da pipeline e ao menor custo operacional. O projeto consiste em receber laudos médicos, processar os textos e classificá-los de acordo com a linha de cuidado requerida para o tratamento.

Como os laudos são documentos em texto livre, a pipeline será responsável por realizar o tratamento e a padronização dessas informações, transformando os dados não estruturados em informações estruturadas que possam ser armazenadas em um banco de dados e posteriormente utilizadas para análise e acompanhamento.

Embora a classificação da linha de cuidado seja relevante durante todo o tratamento do paciente, não existe a necessidade de que uma nova classificação seja disponibilizada imediatamente após a chegada de cada laudo. Dessa forma, não há um requisito de baixa latência que justifique a manutenção de uma API de inferência em tempo real.

A pipeline poderá ser executada periodicamente, por exemplo, em intervalos de uma hora ou uma vez ao dia, processando os novos laudos disponíveis desde a última execução. Essa abordagem permite manter um registro contínuo das classificações, sem a necessidade de manter uma infraestrutura de inferência ativa constantemente.

Além de reduzir a complexidade operacional, o processamento Batch permite utilizar os recursos computacionais apenas durante as execuções da pipeline, contribuindo para a redução dos custos de infraestrutura.

### Escolha da nuvem

Para a infraestrutura em nuvem, foi escolhida a AWS (Amazon Web Services) devido à disponibilidade de serviços gerenciados que atendem às diferentes etapas da pipeline de Machine Learning. A plataforma permite centralizar o armazenamento dos laudos e dos dados processados, executar os processos de tratamento e inferência e armazenar os resultados, sem a necessidade de manter toda a infraestrutura física localmente.

A escolha também está alinhada à utilização de processamento Batch. Como a pipeline não precisa estar disponível continuamente, os recursos computacionais podem ser provisionados e utilizados durante as execuções programadas, evitando a necessidade de manter uma infraestrutura de processamento ativa permanentemente.

Além disso, a AWS oferece serviços específicos para armazenamento, processamento e Machine Learning, permitindo que a arquitetura seja escalada conforme o volume de laudos aumente. Dessa forma, a solução pode começar com uma infraestrutura simples e evoluir conforme as necessidades do projeto, sem exigir uma mudança completa da arquitetura.

Portanto, a AWS foi escolhida principalmente pela combinação de serviços gerenciados, flexibilidade, escalabilidade e compatibilidade com o processamento Batch definido para a pipeline.

## Bibliotecas usadas
- Fast API
- Prometheus
- Airflow
- SciKit-Learn
- Skl2onnx
- Onnx
- Onnxruntime

## Dados
    https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus

