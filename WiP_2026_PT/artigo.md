# SenseCV: Uma Base de Dados Visual--Inercial Egocêntrica e Multilocal para Navegação Assistiva em Ambientes Internos

**Autores:** André Luis Albuquerque Pinheiro, Isaque Efraim Martins Oliveira e Elias Teodoro da Silva Jr.

## Resumo



## Palavras-chave



## Introdução

A deficiência visual pode comprometer a autonomia em atividades de orientação e locomoção, especialmente em ambientes desconhecidos, dinâmicos ou sujeitos a alterações. A Organização Mundial da Saúde estima que ao menos 2,2 bilhões de pessoas apresentam algum grau de deficiência visual para perto ou para longe \cite{who2026}. Para essa população, obstáculos no nível do solo ou suspensos, corredores estreitos, circulação de pessoas e mudanças na organização do ambiente podem tornar o deslocamento interno uma tarefa complexa e aumentar a dependência de auxílio externo.

Avanços em visão computacional e inteligência artificial têm impulsionado o desenvolvimento de tecnologias assistivas capazes de reconhecer caminhos livres, detectar obstáculos, estimar a posição do usuário e produzir instruções de navegação \cite{casanova2025, khan2022}. Entretanto, muitas soluções dependem de câmeras RGB-D, LiDAR, sensores ultrassônicos, dispositivos vestíveis personalizados ou unidades externas de processamento. Embora esses recursos forneçam informações detalhadas sobre o ambiente, eles também podem aumentar o custo, o peso e a complexidade de implantação do sistema \cite{gao2025}.

Os smartphones constituem uma alternativa atraente por reunirem câmera, sensores inerciais, capacidade de processamento e armazenamento em um único dispositivo amplamente disponível. Além disso, uma solução pode utilizar os sensores durante a construção da base e depender somente da câmera na etapa de inferência. Essa estratégia reduz os requisitos para uma futura implantação e permite aproveitar os sinais inerciais como fonte de supervisão. O AI Guide Dog, por exemplo, emprega vídeos egocêntricos de smartphone para prever direções viáveis de navegação, utilizando os sensores móveis durante a coleta e a geração dos rótulos \cite{jadhav2025}. De forma relacionada, \cite{medeiros2021} utilizaram imagens, acelerômetro e giroscópio de smartphones para modelar movimentos frontais e laterais associados ao desvio de obstáculos em ambientes internos.

Apesar desses resultados, a construção e a avaliação das bases continuam sendo desafios centrais. Gravações egocêntricas apresentam elevada redundância temporal, pois quadros consecutivos frequentemente exibem os mesmos corredores, obstáculos e pontos de vista. Em bases constituídas por quadros extraídos de vídeos, a divisão aleatória pode distribuir imagens com alta correlação espaço-temporal entre os subconjuntos de treinamento, validação e teste, causando vazamento de informação e superestimando o desempenho dos modelos \cite{figueiredo2024leakage}. Estudos sobre sistemas de alerta de colisão também mostram que diferenças de qualidade visual, origem ou composição das classes podem ser exploradas como atalhos pelos modelos, sem que o fenômeno de interesse seja efetivamente aprendido \cite{chitraranjan2025}. Além disso, bases obtidas a partir de caminhadas naturais tendem a apresentar predominância da ação frontal, enquanto ações importantes para a segurança, como desviar e parar, possuem menor representação \cite{jadhav2025}.

Este artigo apresenta o *SenseCV*, uma base visual--inercial egocêntrica, coletada em múltiplos ambientes internos e desenvolvida para apoiar pesquisas em navegação local e desvio de obstáculos. A proposta combina um aplicativo próprio para aquisição sincronizada de vídeo e sensores, um protocolo controlado e centrado em eventos e uma estratégia de particionamento que preserva a independência entre as cenas utilizadas no treinamento, na validação e no teste. As principais contribuições deste trabalho são:
- um aplicativo Android de código aberto que registra, de forma sincronizada, vídeo, metadados do dispositivo e diferentes dados de sensores internos;
- uma base coletada em três instituições, contendo eventos controlados de caminho livre, desvio lateral e parada diante de obstáculos;
- um esquema de rotulagem que combina as anotações do protocolo com os sinais do acelerômetro e do giroscópio;
- uma estratégia de particionamento por gravação, cena e local, evitando que quadros temporalmente correlacionados sejam distribuídos entre diferentes subconjuntos; e
- um protocolo experimental com um modelo de duas cabeças, no qual a ação
  local é obtida pela combinação entre a presença de obstáculo e a direção do
  desvio.


O restante deste artigo está organizado da seguinte forma. A Seção II apresenta os trabalhos relacionados sobre navegação assistiva, visão egocêntrica e bases visual--inerciais. A Seção III descreve a metodologia, incluindo o aplicativo SenseCV, o protocolo de coleta, o esquema de rotulagem, a formulação do problema, as estatísticas da base, o particionamento e as métricas de avaliação. A Seção IV apresenta os resultados experimentais. Por fim, a Seção V reúne as conclusões e indica as próximas etapas da pesquisa.


## Trabalhos Relacionados

A navegação visual em ambientes internos abrange diferentes problemas, como localização, planejamento de rotas, reconhecimento de caminhos livres, detecção de obstáculos e geração de instruções. Revisões da área mostram que soluções baseadas em câmeras apresentam vantagens relacionadas ao custo e à disponibilidade dos dispositivos, mas ainda enfrentam desafios de iluminação, oclusão, custo computacional, posicionamento da câmera e generalização entre ambientes \cite{khan2022}. Nesse contexto, os trabalhos mais próximos ao *SenseCV* podem ser organizados em soluções móveis para percepção de obstáculos, métodos de predição egocêntrica, bases visual--inerciais e conjuntos de dados originalmente desenvolvidos para plataformas robóticas.

\cite{breve2020} propuseram uma solução orientada a smartphones que classifica imagens como caminho livre ou não livre, demonstrando a viabilidade de utilizar transferência de aprendizado em uma base relativamente pequena, composta por cenas internas e externas. Em uma formulação baseada na identificação explícita dos objetos, \cite{bastidas2025detectra} apresentaram a base Detectra, formada por 7.600 imagens e 76 classes de obstáculos coletadas em uma instituição de ensino para pessoas com deficiência visual. Um modelo YOLOv5s foi integrado a uma aplicação móvel com retorno auditivo e háptico. Entretanto, as imagens representam objetos isolados ou composições de objetos e não sequências egocêntricas associadas à ação de navegação que deve ser executada pelo usuário.

Soluções vestíveis podem incorporar informações mais completas sobre o ambiente, embora exijam hardware adicional. \cite{gao2025} desenvolveram o WOAD, que combina câmera infravermelha, sensor de profundidade, processamento em FPGA, smartphone e retorno auditivo e tátil. O sistema estima características como distância, direção e movimento dos obstáculos, apresentando elevada capacidade de desvio em experimentos internos e externos. Essa abordagem evidencia os benefícios da percepção multimodal, mas também contrasta com propostas baseadas exclusivamente em sensores presentes em smartphones comuns.

A predição egocêntrica da direção de caminhada é particularmente próxima ao objetivo do *SenseCV*. \cite{jadhav2025} formularam a navegação como uma classificação multirrótulo de frente, esquerda e direita a partir de vídeos capturados por um smartphone posicionado no tórax. Os sensores móveis são utilizados na geração dos rótulos, enquanto a inferência final depende apenas das imagens. A base possui aproximadamente 57 horas e foi particionada de modo a evitar sobreposição de cenas. Interseções internas, nas quais mais de uma direção era possível, receberam correções manuais. Apesar da diversidade da base, o trabalho não contempla uma classe de parada e reconhece dificuldades em cenas dinâmicas, caminhos bloqueados e estruturas ambíguas.

\cite{medeiros2021} também utilizaram imagens de smartphone e sinais inerciais para estudar o desvio de obstáculos em ambientes internos. A tarefa foi organizada em duas classificações, com cinco níveis de velocidade frontal e cinco níveis de velocidade lateral, avaliando arquiteturas como VGG16, ResNet50 e DroNet. O trabalho demonstra a possibilidade de empregar o movimento medido pelos sensores como supervisão das imagens. Entretanto, a divisão dos dados é descrita no nível das imagens, sem explicitar o agrupamento por gravação, percurso ou ambiente. O *SenseCV* mantém a aquisição visual--inercial, mas utiliza ações locais discretas e preserva as gravações completas durante o particionamento.

\cite{plikynas2022} apresentaram uma proposta de navegação interna baseada em rotas coletadas por voluntários, câmeras RGB e de profundidade, IMU, SLAM, processamento em nuvem e retorno auditivo e tátil. Um aspecto relevante é o emparelhamento automático entre imagens e classes de movimento extraídas da IMU. Os autores utilizaram uma ConvLSTM para classificar os movimentos frente, esquerda e direita, além de estimar a confiabilidade da predição. Embora essa estratégia seja próxima à rotulagem visual--inercial, o sistema depende de equipamento vestível especializado, rotas previamente percorridas e infraestrutura computacional adicional. O *SenseCV*, por sua vez, concentra-se na disponibilização de dados capturados com smartphones comuns e em decisões locais que não dependem de uma rota previamente mapeada.

O aprendizado multitarefa fornece outra formulação relevante. \cite{loquercio2018} propuseram a DroNet, uma rede com extrator visual compartilhado e duas saídas: um valor contínuo de direção e uma probabilidade de colisão. \cite{chang2022} adaptaram essa formulação à base HDIN, coletada com VANTs em ambientes internos reais, e discutiram problemas como oscilações dos sensores, unidades incompatíveis e ruído nos rótulos contínuos. Embora as plataformas e os comandos de controle sejam distintos daqueles utilizados por pedestres, esses trabalhos motivam a comparação entre a predição direta de uma ação e o aprendizado conjunto da ação com uma representação auxiliar de obstáculo.

Bases capturadas por smartphones também são relevantes para tarefas diferentes da recomendação de direção. \cite{samadzadeh2024} apresentaram a AUT-VI, composta por 126 sequências capturadas em 17 locais, contendo vídeo, IMU e GPS sincronizados. A base inclui objetos dinâmicos, reflexos, mudanças de iluminação, movimentos bruscos e diferentes formas de segurar o aparelho, além de disponibilizar um aplicativo para reprodução do processo de coleta. Seu objetivo principal, entretanto, é a avaliação de algoritmos de odometria visual--inercial e estimação de pose, e não a escolha de uma ação local diante de obstáculos.

Conjuntos robóticos oferecem maior diversidade sensorial e de interações, mas não representam diretamente a dinâmica de uma pessoa com deficiência visual. \cite{karnan2022scand} apresentaram o SCAND, com 8,7 horas e 138 trajetórias coletadas por meio dos robôs Boston Dynamics Spot e Clearpath Jackal em situações sociais internas e externas. A base contém câmeras, LiDAR, IMU, odometria e comandos de joystick, além de interações com pessoas, multidões e veículos. Os robôs podem atingir velocidades de 1,6 e 2,0~m/s, respectivamente. Como referência, \cite{liu2019travel} consideram uma velocidade média de aproximadamente 0,72~m/s para a caminhada de pessoas com deficiência visual. Além da diferença de velocidade, a altura da câmera, a morfologia, a dinâmica e os comandos dos robôs limitam a aplicação direta do SCAND ao contexto assistivo humano.

\cite{yan2020flobot} apresentaram uma base coletada por um robô autônomo de limpeza em aeroporto, armazém e supermercado. O FLOBOT utiliza LiDAR 2D e 3D, câmeras RGB-D e câmera estéreo para detectar pessoas, objetos no chão e alterações do ambiente. Os experimentos mostram que modelos treinados em um local podem perder desempenho quando avaliados em outro, reforçando a importância da diversidade ambiental. Apesar dessa contribuição, a plataforma industrial, a quantidade de sensores e o objetivo de limpeza autônoma diferem da perspectiva egocêntrica e dos recursos disponíveis em um smartphone.

A Tabela~\ref{tab:related_comparison} resume os trabalhos mais próximos em relação à plataforma, ao ambiente, à tarefa e ao particionamento.

\begin{table*}[!t]
\centering
\caption{Comparação entre bases para navegação egocêntrica e assistiva.}
\label{tab:related_comparison}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{5pt}
\scriptsize
\begin{tabular}{
p{2.4cm}
p{0.8cm}
p{2.4cm}
p{1.8cm}
p{5.1cm}
p{2.4cm}}
\hline
**Trabalho** &
**Ano** &
**Plataforma** &
**Ambiente** &
**Tarefa** &
**Particionamento** \\
\hline

Breve e Fischer \cite{breve2020} &
2020 &
Smartphone &
Interno e externo &
Classificação de caminho livre ou obstruído &
Não informado \\

Medeiros *et al.* \cite{medeiros2021} &
2021 &
Smartphone &
Interno &
Classificação de velocidade frontal e lateral &
Por imagens \\

AIGD \cite{jadhav2025} &
2025 &
Smartphone &
Interno e externo &
Predição das direções frente, esquerda e direita &
Cenas disjuntas \\

Plikynas *et al.* \cite{plikynas2022} &
2022 &
Dispositivo vestível &
Interno &
Predição das direções frente, esquerda e direita &
Por rotas \\

AUT-VI \cite{samadzadeh2024} &
2024 &
Smartphone &
Interno e externo &
Odometria visual--inercial &
Por sequências \\

Detectra \cite{bastidas2025detectra} &
2025 &
Smartphone &
Interno e externo &
Detecção de classes de obstáculos &
Por imagens \\

**SenseCV** &
**2026** &
**Smartphone** &
**Interno e multilocal** &
**Presença de obstáculo e ação local** &
**Por gravações e locais** \\

\hline
\end{tabular}
\end{table*}

Conforme sintetizado na Tabela~\ref{tab:related_comparison}, os trabalhos existentes apresentam contribuições importantes para detecção de obstáculos, predição egocêntrica, localização e aquisição visual--inercial. Entretanto, permanecem limitadas as bases que combinam aquisição por smartphone comum, vídeo egocêntrico sincronizado com sensores inerciais, múltiplos ambientes internos, eventos controlados de desvio e parada e particionamento sem sobreposição de cenas. O *SenseCV* busca preencher essa lacuna ao disponibilizar os dados brutos do conjunto de teste, o aplicativo de aquisição e um protocolo reproduzível para estudar decisões locais de navegação e a capacidade de generalização dos modelos.



## Metodologia

A metodologia foi definida em torno de dois objetivos. Primeiro, a base deve expor os modelos a diferentes edifícios, corredores, pontos de vista, objetos e condições de iluminação. Segundo, a organização temporal dos dados deve ser preservada tanto na rotulagem quanto no particionamento. Os modelos baseados em imagens isoladas podem ser treinados e avaliados quadro a quadro, enquanto os modelos temporais utilizam sequências de quadros. Entretanto, em ambos os casos, todos os quadros pertencentes a uma mesma gravação ou evento devem permanecer no mesmo subconjunto, evitando que imagens consecutivas ou quase idênticas sejam distribuídas entre treino, validação e teste.

### Aplicativo de Coleta SenseCV

A aquisição dos dados foi realizada com o aplicativo Android de código aberto *SenseCV*.\footnote{Repositório público contendo o código-fonte, as instruções de instalação e a documentação de uso do aplicativo: \url{https://github.com/SenseCV}.} Para cada gravação, o aplicativo armazena o vídeo da câmera, os timestamps associados aos quadros, os metadados do dispositivo e da configuração da câmera, além de arquivos no formato JavaScript Object Notation (JSON) contendo os dados dos sensores disponibilizados pelo aparelho. As modalidades registradas incluem acelerômetro, giroscópio e magnetômetro. Quando o serviço de localização está disponível, o aplicativo também coleta as coordenadas de latitude e longitude, seus respectivos timestamps e informações sobre o estado do GPS.

É importante distinguir as modalidades *capturadas* daquelas *utilizadas atualmente*. Embora o aplicativo armazene diferentes fluxos sensoriais para ampliar as possibilidades de reutilização da base, somente os dados do acelerômetro e do giroscópio são empregados no esquema de rotulagem adotado neste trabalho. Os dados de GPS e do magnetômetro são mantidos como informações complementares para estudos futuros, como análise de trajetórias, localização e fusão multimodal. Cada sessão é organizada em uma estrutura própria de arquivos no dispositivo, o que facilita a revisão, a seleção e, quando necessário, a anonimização das gravações antes de sua transferência ou disponibilização.

O arquivo de metadados registra informações como fabricante e modelo do dispositivo, versão do Android, câmera selecionada, distância focal, orientação, resolução, taxa de quadros e configurações da câmera. O usuário pode selecionar a lente, a orientação em modo retrato ou paisagem, o modo quadrado, a resolução e uma taxa nominal de 15, 30 ou 60~FPS. A gravação também pode ser iniciada e interrompida por meio de um controle Bluetooth externo, reduzindo a necessidade de interação direta com a tela e a movimentação provocada pelo toque no aparelho.

### Protocolo de Coleta

As gravações foram realizadas em Fortaleza, Brasil, no Instituto Federal do Ceará (IFCE), na Universidade Federal do Ceará (UFC) e na Biblioteca Pública Estadual do Ceará (BECE). A versão atual contém 13 sessões coletadas entre junho e julho de 2026. A maioria das sessões utilizou um Samsung Galaxy A56, com lente traseira de 1,74~mm e vídeo quadrado de $480\times480$; uma sessão inicial utilizou um POCO X3 Pro em $720\times720$.

Em testes preliminares, observou-se que a fixação direta do smartphone ao corpo introduzia oscilações excessivas decorrentes da caminhada, afetando tanto a estabilidade do vídeo quanto os sinais inerciais. Para reduzir essas interferências e padronizar a orientação da câmera, as coletas foram realizadas com o**ajustar aqui, deixar mais claro** gimbal S5 CamfreeGo. O estabilizador diminui movimentos involuntários associados ao balanço corporal e permite distinguir com maior clareza as variações relacionadas às ações de seguir em frente, desviar e parar, aspecto particularmente relevante porque o acelerômetro e o giroscópio são empregados no esquema de rotulagem da base. **ajustar isso aqui** Em parte das coletas, foram utilizadas duas alturas nominais, aproximadamente 50~cm e 100~cm em relação ao piso, para introduzir variação no ponto de vista. **[PREENCHER: confirmar o número total de portadores e descrever como o gimbal foi transportado e mantido nas alturas definidas]**

O protocolo é centrado em eventos completos de navegação, e não em amostragem oportunista de quadros. Nos eventos de desvio, o portador inicia aproximadamente 5~m antes do obstáculo, começa a manobra a cerca de 1,5~m dele e continua por aproximadamente 1,5~m após ultrapassá-lo. Alguns corredores da BECE exigiram distâncias iniciais menores, entre 3,2 e 3,7~m, devido às limitações físicas do local. Nos eventos de parada, o portador se aproxima a partir de aproximadamente 5~m e para cerca de 1~m antes do obstáculo. Os eventos de caminho livre consistem em caminhada contínua sem obstáculo planejado, geralmente por dez ou mais passos.

O protocolo registra a posição do obstáculo, o desvio solicitado, o corredor, a orientação do celular e uma descrição textual do evento. Foram incluídos objetos estáticos comuns, como lixeiras, cones, placas de sinalização, caixas, cadeiras, extintores, escadas e livros, além de pessoas em diferentes poses. O desenho experimental produz deliberadamente repetições de desvios para a esquerda e para a direita, bem como sequências de parada e caminhada livre, reduzindo a predominância extrema da ação frontal encontrada em bases egocêntricas coletadas durante caminhadas naturais \cite{jadhav2025}.

Os dados brutos correspondentes exclusivamente ao conjunto de teste estão disponíveis publicamente no Kaggle.\footnote{Página do conjunto de dados, contendo as gravações originais e os arquivos sensoriais associados ao conjunto de teste: \url{https://www.kaggle.com/datasets/sensecv/test-set}.} Os conjuntos de treinamento e validação serão disponibilizados em trabalhos futuros**[PREENCHER: é importante falar isso??]**. A disponibilização inclui os vídeos, os dados dos sensores e os metadados gerados pelo aplicativo, preservando a organização original das sessões.

### Esquema de Rotulagem e Formulação do Problema

Cada evento fornece dois rótulos utilizados simultaneamente pelo modelo. A
primeira cabeça prediz a presença de obstáculo como uma saída binária,

$
    o \in \{0,1\},
$

em que $o=0$ representa **caminho livre** e $o=1$ representa
**obstáculo**. A segunda cabeça prediz a direção do desvio como uma saída
categórica com três valores,

$
    d \in \{0,1,2\},
$

em que $d=0$ representa **desvio para a esquerda**, $d=1$ representa
**desvio para a direita** e $d=2$ representa **sem desvio**. Portanto,
**frente** e **parar** não são classes diretamente produzidas pelo modelo, mas ação local final é decodificada pela combinação das duas saídas:

| Saída da cabeça de obstáculo | Saída da cabeça de desvio | Ação final |
|---|---|---|
| `0` — caminho livre | `0` — esquerda | **esquerda** |
| `0` — caminho livre | `1` — direita | **direita** |
| `0` — caminho livre | `2` — sem desvio | **frente** |
| `1` — obstáculo | `0` — esquerda | **esquerda** |
| `1` — obstáculo | `1` — direita | **direita** |
| `1` — obstáculo | `2` — sem desvio | **parar** |

As combinações `(0,0)` e `(0,1)` também são válidas. Mesmo quando a primeira
cabeça indica caminho livre, a segunda pode recomendar um movimento para a
esquerda ou para a direita a fim de conduzir a pessoa para uma região mais
central do corredor. Assim, a cabeça de obstáculo altera a ação final somente
quando a cabeça de desvio produz `2` — sem desvio: caminho livre resulta em
**frente**, enquanto obstáculo resulta em **parar**.

Essa codificação impede que o lado do obstáculo seja confundido com a ação
executada. Um obstáculo localizado à direita, por exemplo, pode exigir um
desvio para a esquerda; o rótulo da segunda cabeça descreve a direção da
manobra, não a posição do objeto.

#### Seleção automática baseada no acelerômetro e no giroscópio

O procedimento de seleção automática parte da sincronização das séries
inerciais com a sequência visual. As medições do acelerômetro e do giroscópio
são interpoladas nos instantes associados aos quadros, produzindo uma
representação multimodal em uma base temporal comum. Essa etapa permite
relacionar alterações do movimento corporal ao conteúdo visual observado
imediatamente antes e durante cada ação.

Para reduzir a dependência da orientação física do telefone, os sinais são
transformados do sistema de coordenadas do dispositivo para uma base definida
pela gravidade e pela direção frontal da câmera. Seja
\(\hat{\mathbf{u}}\) o vetor unitário vertical estimado a partir da componente
de baixa frequência da aceleração. A direção frontal
\(\hat{\mathbf{f}}\) é obtida pela projeção do eixo óptico no plano ortogonal a
\(\hat{\mathbf{u}}\), e a direção lateral é definida por

\[
\hat{\mathbf{l}}=
\hat{\mathbf{u}}\times\hat{\mathbf{f}}.
\]

Essa transformação fornece uma descrição aproximadamente invariável à atitude
do dispositivo e permite interpretar o movimento em termos de componentes
vertical, longitudinal e lateral.

Dois descritores complementares são então calculados. O primeiro corresponde à
componente da velocidade angular em torno do eixo vertical, obtida pela
projeção do vetor medido pelo giroscópio sobre
\(\hat{\mathbf{u}}\). O segundo representa o movimento lateral estimado a
partir do acelerômetro: a contribuição gravitacional é removida, a aceleração
linear é projetada sobre \(\hat{\mathbf{l}}\) e integrada temporalmente. Como a
integração amplifica erros de polarização e componentes de baixa frequência,
aplica-se uma correção local de deriva e uma etapa de suavização. O sinal
resultante é utilizado como descritor cinemático relativo, e não como uma
estimativa absoluta da velocidade de deslocamento.

A escolha entre os dois descritores é realizada separadamente para cada clipe.
Quando a componente angular apresenta informação dinâmica suficiente, ela é
utilizada para localizar mudanças de direção. Quando essa componente é
atenuada pela estabilização da câmera, pela posição do dispositivo ou pela
dinâmica da captura, o sistema emprega a componente lateral derivada da
aceleração. Desse modo, a seleção automática procura utilizar a modalidade
inercial que melhor representa o movimento observado, sem pressupor que um
único sensor seja igualmente informativo em todas as gravações.

A segmentação temporal considera conjuntamente magnitude, duração e
persistência dos sinais. Variações breves e isoladas são tratadas como ruído,
enquanto segmentos temporalmente consistentes são associados a possíveis
mudanças de trajetória. Segmentos consecutivos de polaridades opostas podem
representar fases complementares de uma mesma manobra de contorno e, por isso,
são analisados como um único evento. A direção atribuída ao evento é definida
pelo sinal da componente dominante no sistema de coordenadas alinhado à
gravidade.

Os estados de caminhada, desaceleração e parada são caracterizados por uma
medida local da variabilidade da magnitude da aceleração. Durante a marcha,
essa medida apresenta estrutura aproximadamente periódica em razão da
alternância das passadas. A redução progressiva de sua energia indica uma
transição de velocidade, enquanto a manutenção de baixa variabilidade é
compatível com um estado estacionário. A autocorrelação do sinal é utilizada
como evidência adicional da regularidade da marcha e também como um critério de
qualidade da captura.

Para cada evento identificado, o método distingue o intervalo da ação
\([T_1,T_2]\) e um intervalo anterior de decisão
\([T_1-\Delta,T_1]\). Também pode ser utilizado um intervalo expandido, que
inclui margens antes e depois da ação para acomodar incertezas de sincronização
e de delimitação temporal. Para o treinamento visual, prioriza-se o intervalo
de decisão, pois ele contém as observações disponíveis antes da resposta
motora do participante.

A confiabilidade da seleção é avaliada a partir da regularidade da amostragem,
da presença de lacunas, da saturação dos sensores, da periodicidade da marcha,
da separabilidade entre estados de movimento e da intensidade relativa do
evento. Capturas inadequadas são descartadas, enquanto casos ambíguos
permanecem disponíveis para revisão manual com uma indicação de menor
confiança. A seleção inercial determina principalmente a localização temporal
da reação; o significado visual da cena e a classe final continuam vinculados
à anotação supervisionada do protocolo.

Em termos gerais, a formulação adotada é compatível com métodos consolidados de
processamento inercial para navegação pedestre, análise da marcha, detecção de
estados de movimento e controle de deriva
\cite{manos2019gravity,moenilssen2004gait,park2010zerovelocity,munozdiaz2017drift}.

Dada uma entrada visual $X_t$, representada por um quadro ou por um clipe curto
encerrado no instante $t$, o modelo com extrator compartilhado é definido por

$
    f_{\theta}(X_t)=\left(\hat{o}_t,\hat{d}_t\right),
$

em que $\hat{o}_t$ é a saída binária da cabeça de obstáculo e
$\hat{d}_t$ é a saída de três classes da cabeça de desvio. A ação apresentada
ao usuário é então

$
    \hat{y}_t=g\left(\hat{o}_t,\hat{d}_t\right),
$

onde $g$ aplica a tabela de decodificação anterior. Assim, o modelo aprende
conjuntamente a presença de obstáculo e a direção da manobra, seguindo a
motivação multitarefa de DroNet e HDIN \cite{loquercio2018,chang2022}.
Nesta etapa, **parar** representa a resposta de parada demonstrada no
protocolo, e não a comprovação de que inexiste um caminho alternativo fora do
campo de visão da câmera.

### Estatísticas da Base e Particionamento

Após a validação das gravações, a versão atual da base reúne 592 eventos de navegação, distribuídos em 13 sessões e com duração total de 1~h, 45~min e 42~s. O IFCE concentra 320 eventos provenientes de seis sessões, a BECE contém 95 eventos de duas sessões e a UFC reúne 177 eventos de cinco sessões.

Após a combinação dos dois rótulos, todos os eventos possuem uma ação local
definida: 180 pertencem à classe frente, 197 à classe esquerda, 187 à classe
direita e 28 à classe parar. Para a cabeça binária de presença de obstáculo,
142 eventos representam caminho livre e 450 contêm algum obstáculo. Na cabeça
de desvio, frente e parar compartilham o rótulo `2` — sem desvio — e são
diferenciados pela saída da cabeça de obstáculo. Esses valores correspondem ao
nível dos eventos completos de gravação; a quantidade de quadros ou clipes
utilizados pelos modelos depende da política de amostragem temporal adotada. A
Tabela~\ref{tab:stats} resume a distribuição das ações decodificadas e da
presença de obstáculo.

\begin{table}[t]
\caption{Distribuição dos rótulos no nível dos eventos válidos.}
\label{tab:stats}
\centering
\scriptsize
\begin{tabular}{lrr}
\toprule
Família & Classe & Eventos \\
\midrule
Ação & Frente & 180 \\
     & Esquerda & 197 \\
     & Direita & 187 \\
     & Parar & 28 \\
\midrule
Obstáculo & Livre & 142 \\
          & Obstáculo & 450 \\
\bottomrule
\end{tabular}
\end{table}

O particionamento é definido no nível das gravações, antes da extração de quadros ou da formação de clipes. Cada vídeo de origem e todas as amostras derivadas dele permanecem em um único subconjunto. Assim, modelos que realizam classificação quadro a quadro podem utilizar imagens individuais, enquanto modelos temporais podem receber sequências de quadros, sem que amostras provenientes da mesma gravação sejam distribuídas entre treino, validação e teste.

Além disso, gravações correlacionadas de um mesmo corredor ou configuração de cena são mantidas no mesmo subconjunto. Essa regra reduz a possibilidade de que o modelo seja avaliado em imagens quase idênticas às observadas durante o treinamento. A avaliação considera uma divisão sem sobreposição de cenas e um protocolo entre locais, no qual os modelos são treinados com dados de duas instituições e avaliados na terceira. **[PREENCHER: inserir a composição final dos conjuntos de treino, validação e teste]**

Essa estratégia é mais rigorosa do que a divisão aleatória por quadros. Em \cite{medeiros2021}, as imagens foram reorganizadas aleatoriamente e divididas em treinamento, validação e teste, sem que fosse explicitada uma separação por gravação ou cena. Em bases constituídas por quadros extraídos de vídeos, a divisão aleatória pode distribuir imagens com alta correlação espaço-temporal entre os subconjuntos, causando vazamento de informação e superestimando o desempenho dos modelos \cite{figueiredo2024leakage}. No *SenseCV*, todas as amostras derivadas de uma mesma gravação permanecem juntas, e a avaliação entre locais busca fornecer uma estimativa mais realista da capacidade de generalização para corredores, objetos e ambientes não observados durante o treinamento.

### Métricas de Avaliação

O desempenho do modelo é avaliado por meio da acurácia e do *F1-score*. Essas
métricas são calculadas separadamente para a cabeça binária de obstáculo e
para a cabeça de desvio com três classes. A ação final decodificada também é
avaliada por uma matriz de confusão com as classes frente, esquerda, direita e
parar. A acurácia representa a proporção total de amostras classificadas
corretamente, enquanto o *F1 score* combina precisão e revocação em uma única
medida \cite{rainio2024metrics}.

Seja $C$ o número de classes e $\mathbf{M}\in\mathbb{N}^{C\times C}$ a matriz de confusão, em que $M_{ij}$ representa o número de amostras cuja classe real é $i$ e cuja classe prevista é $j$. A acurácia é definida por

$
\mathrm{Acc} =
\frac{\displaystyle\sum_{i=1}^{C} M_{ii}}
{\displaystyle\sum_{i=1}^{C}\sum_{j=1}^{C} M_{ij}}.
\label{eq:accuracy}
$

Para cada classe $i$, a precisão e a revocação são calculadas,
respectivamente, por

$
\mathrm{Prec}_{i} =
\frac{M_{ii}}
{\displaystyle\sum_{k=1}^{C}M_{ki}},
\qquad
\mathrm{Rec}_{i} =
\frac{M_{ii}}
{\displaystyle\sum_{k=1}^{C}M_{ik}}.
\label{eq:precision_recall}
$

O *F1-score* da classe $i$ corresponde à média harmônica entre precisão e
revocação:

$
F_{1,i} =
2\frac{\mathrm{Prec}_{i}\mathrm{Rec}_{i}}
{\mathrm{Prec}_{i}+\mathrm{Rec}_{i}}.
\label{eq:f1_class}
$

Como as classes da base apresentam frequências distintas, especialmente pela menor quantidade de eventos da classe parar, os resultados são reportados utilizando a macro-F1, definida como a média não ponderada dos valores obtidos para cada classe:

$
\mathrm{Macro}\text{-}F_{1} =
\frac{1}{C}\sum_{i=1}^{C}F_{1,i}.
\label{eq:macro_f1}
$

Dessa forma, todas as classes contribuem igualmente para a métrica final, independentemente de sua frequência. A acurácia é apresentada como uma medida global complementar, enquanto a macro-F1 é utilizada como principal indicador para a comparação entre os modelos \cite{farhadpour2024metrics}.

### Configuração Experimental
**[PREENCHER: isaque]**

## Resultados


## Conclusão
Este artigo apresentou a primeira versão do *SenseCV*, uma base visual--inercial egocêntrica e multilocal para navegação assistiva local em ambientes internos. Um aplicativo Android dedicado registra vídeo, metadados e múltiplos dados de sensores do smartphone de forma sincronizada. O método atual de rotulagem utiliza acelerômetro e giroscópio, enquanto as demais modalidades são preservadas para pesquisas futuras. 

\begin{thebibliography}{99}

\bibitem{who2026}
World Health Organization,
``Blindness and vision impairment,''
Feb. 10, 2026. [Online]. Available:
\url{https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment}.
[Accessed: Jul. 27, 2026].

\bibitem{casanova2025}
E. Casanova, D. Guffanti, and L. Hidalgo,
``Technological advancements in human navigation for the visually
impaired: A systematic review,''
*Sensors*, vol. 25, no. 7, Art. no. 2213, 2025,
doi: 10.3390/s25072213.

\bibitem{khan2022}
D. Khan, Z. Cheng, H. Uchiyama, S. Ali, M. Asshad, and
K. Kiyokawa,
``Recent advances in vision-based indoor navigation: A systematic
literature review,''
*Computers \& Graphics*, vol. 104, pp. 24--45, 2022,
doi: 10.1016/j.cag.2022.03.005.

\bibitem{gao2025}
Y. Gao *et al.*,
``A wearable obstacle avoidance device for visually impaired
individuals with cross-modal learning,''
*Nature Communications*, vol. 16, Art. no. 2857, 2025,
doi: 10.1038/s41467-025-58085-x.

\bibitem{jadhav2025}
A. Jadhav *et al.*,
``AI Guide Dog: Egocentric path prediction on smartphone,''
*Proceedings of the AAAI Symposium Series*,
vol. 5, no. 1, pp. 220--227, 2025,
doi: 10.1609/aaaiss.v5i1.35591.

\bibitem{medeiros2021}
D. S. de Medeiros, T. H. Araújo, E. T. da Silva Júnior,
and G. L. B. Ramalho,
``Using images to avoid collisions and bypass obstacles in indoor
environments,''
in *Anais Estendidos da XXXIV Conference on Graphics,
Patterns and Images*, 2021,
doi: 10.5753/sibgrapi.est.2021.20030.

\bibitem{figueiredo2024leakage}
R. B. D. Figueiredo and H. A. Mendes,
``Analyzing information leakage on video object detection datasets
by splitting images into clusters with high spatiotemporal
correlation,''
*IEEE Access*, vol. 12, pp. 47646--47655, 2024,
doi: 10.1109/ACCESS.2024.3383047.

\bibitem{chitraranjan2025}
C. Chitraranjan, V. Vipulananthan, and T. Sritharan,
``Vision-based collision warning systems with deep learning:
A systematic review,''
*Journal of Imaging*, vol. 11, no. 2, Art. no. 64, 2025,
doi: 10.3390/jimaging11020064.

\bibitem{breve2020}
F. A. Breve and C. N. Fischer,
``Visually impaired aid using convolutional neural networks,
transfer learning, and particle competition and cooperation,''
in *Proc. 2020 International Joint Conference on Neural
Networks (IJCNN)*, 2020, pp. 1--8,
doi: 10.1109/IJCNN48605.2020.9207606.

\bibitem{bastidas2025detectra}
G. K. Bastidas-Guacho, M. A. Paguay Alvarado,
P. X. Moreno-Vallejo, P. R. Moreno-Costales,
N. S. Ocaña Yanza, and J. C. Troya Cuestas,
``Computer vision-based obstacle detection mobile system for
visually impaired individuals,''
*Multimodal Technologies and Interaction*,
vol. 9, no. 5, Art. no. 48, 2025,
doi: 10.3390/mti9050048.

\bibitem{plikynas2022}
D. Plikynas, A. Indriulionis, A. Laukaitis, and
L. Sakalauskas,
``Indoor-guided navigation for people who are blind:
Crowdsourcing for route mapping and assistance,''
*Applied Sciences*, vol. 12, no. 1, Art. no. 523, 2022,
doi: 10.3390/app12010523.

\bibitem{loquercio2018}
A. Loquercio, A. I. Maqueda, C. R. del-Blanco, and
D. Scaramuzza,
``DroNet: Learning to fly by driving,''
*IEEE Robotics and Automation Letters*,
vol. 3, no. 2, pp. 1088--1095, 2018,
doi: 10.1109/LRA.2018.2795643.

\bibitem{chang2022}
Y. Chang, Y. Cheng, J. Murray, S. Huang, and G. Shi,
``The HDIN dataset: A real-world indoor UAV dataset with
multi-task labels for visual-based navigation,''
*Drones*, vol. 6, no. 8, Art. no. 202, 2022,
doi: 10.3390/drones6080202.

\bibitem{samadzadeh2024}
A. Samadzadeh, M. H. Mojab, H. Soudani,
S. H. Mireshghollah, and A. Nickabadi,
``Amirkabir campus dataset: Real-world challenges and scenarios
of visual inertial odometry for visually impaired people,''
*arXiv preprint arXiv:2401.03604*, 2024,
doi: 10.48550/arXiv.2401.03604.

\bibitem{karnan2022scand}
H. Karnan *et al.*,
``Socially CompliAnt Navigation Dataset (SCAND):
A large-scale dataset of demonstrations for social navigation,''
*IEEE Robotics and Automation Letters*,
vol. 7, no. 4, pp. 11807--11814, 2022,
doi: 10.1109/LRA.2022.3184025.

\bibitem{liu2019travel}
X. Liu, S. Zhang, J. Zeng, and F. Fan,
``Analysis and optimization strategy of travel system for urban
visually impaired people,''
*Sustainability*, vol. 11, no. 6,
Art. no. 1735, 2019,
doi: 10.3390/su11061735.

\bibitem{yan2020flobot}
Z. Yan, S. Schreiberhuber, G. Halmetschlager,
T. Duckett, M. Vincze, and N. Bellotto,
``Robot perception of static and dynamic objects with an
autonomous floor scrubber,''
*Intelligent Service Robotics*,
vol. 13, no. 3, pp. 403--417, 2020,
doi: 10.1007/s11370-020-00324-9.

\bibitem{rainio2024metrics}
O. Rainio, J. Teuho, and R. Klén,
``Evaluation metrics and statistical tests for machine learning,''
*Scientific Reports*, vol. 14,
Art. no. 6086, 2024,
doi: 10.1038/s41598-024-56706-x.

\bibitem{farhadpour2024metrics}
S. Farhadpour, T. A. Warner, and A. E. Maxwell,
``Selecting and interpreting multiclass loss and accuracy
assessment metrics for classifications with class imbalance:
Guidance and best practices,''
*Remote Sensing*, vol. 16, no. 3,
Art. no. 533, 2024,
doi: 10.3390/rs16030533.

\bibitem{manos2019gravity}
A. Manos, I. Klein, and T. Hazan,
``Gravity-Based Methods for Heading Computation in Pedestrian Dead
Reckoning,''
*Sensors*, vol. 19, no. 5, Art. no. 1170, 2019,
doi: 10.3390/s19051170.

\bibitem{moenilssen2004gait}
R. Moe-Nilssen and J. L. Helbostad,
``Estimation of Gait Cycle Characteristics by Trunk Accelerometry,''
*Journal of Biomechanics*, vol. 37, no. 1, pp. 121--126, 2004,
doi: 10.1016/S0021-9290(03)00233-1.

\bibitem{park2010zerovelocity}
S. K. Park and Y. S. Suh,
``A Zero Velocity Detection Algorithm Using Inertial Sensors for Pedestrian
Navigation Systems,''
*Sensors*, vol. 10, no. 10, pp. 9163--9178, 2010,
doi: 10.3390/s101009163.

\bibitem{munozdiaz2017drift}
E. Munoz Diaz, M. Caamano, and F. J. Fuentes-Sanchez,
``Landmark-Based Drift Compensation Algorithm for Inertial Pedestrian
Navigation,''
*Sensors*, vol. 17, no. 7, Art. no. 1555, 2017,
doi: 10.3390/s17071555.

\end{thebibliography}


