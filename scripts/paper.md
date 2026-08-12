\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{multirow}
\usepackage{array}
\usepackage[table,xcdraw]{xcolor}
\usepackage{url}
\usepackage{booktabs}

%\usepackage{multirow}
\newcommand{\sensecv}{SenseCV}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{SenseCV: A Multi-Location Egocentric Visual--Inertial Dataset for Assistive Navigation in Indoor Environments}

% \title{SenseCV: Uma Base de Dados Visual--Inercial Egocêntrica e Multilocal para Navegação Assistiva em Ambientes Internos\\}


% \author{\IEEEauthorblockN{André Luis Albuquerque Pinheiro, Isaque Efraim Martins Oliveira e Elias Teodoro da Silva Jr.}
% \IEEEauthorblockA{Departamento de Ciência da Computação -- Instituto Federal do Ceará (IFCE), Fortaleza, CE, Brasil\\
% \{andre.luis.albuquerque07, isaque.efraim94\}@aluno.ifce.edu.br, elias@ifce.edu.br}}

\maketitle

\begin{abstract}
Assistive navigation in indoor environments requires models capable of recognizing traversable paths and predicting local actions under different visual conditions. However, video-derived datasets may exhibit high temporal redundancy and information leakage when correlated frames are distributed across the training and test sets. This work presents \sensecv{}, a multi-location egocentric visual--inertial dataset for assistive navigation in indoor environments. The dataset contains 592 events collected across three locations, including free-path navigation, left and right deviations, and stopping events. The data were recorded using smartphones through an open-source Android application that synchronizes videos, metadata, and inertial signals. Accelerometer and gyroscope data are used to temporally delimit the actions and verify label consistency. The evaluation adopts a multitask formulation to classify path status and deviation direction. With one location reserved exclusively for testing, MobileNetV4 achieved 95.58\% accuracy for path-status classification and 89.97\% for deviation classification. The experiments also showed that reducing the sampling rate from 30 to 10~FPS preserves performance, whereas random frame-level splitting introduces information leakage due to the high temporal correlation among samples. These results reinforce the need for partitioning protocols that preserve the separation among recordings, scenes, and locations.
\end{abstract}


% \begin{IEEEkeywords}

% \end{IEEEkeywords}

\section{Introduction} \label{sec:intro}

Visual impairment can compromise autonomy in orientation and mobility activities, particularly in unfamiliar or changing environments. The World Health Organization estimates that at least 2.2 billion people have a near or distance vision impairment \cite{who2026}. In indoor environments, ground-level or overhead obstacles, narrow corridors, pedestrian traffic, and changes in spatial organization make navigation more challenging and may increase dependence on external assistance.

Advances in computer vision and artificial intelligence have enabled the recognition of traversable paths, obstacle detection, user localization, and the generation of navigation instructions \cite{casanova2025,khan2022}. However, many solutions rely on RGB-D cameras, LiDAR, ultrasonic sensors, custom wearable devices, or external processing units. Although these resources provide detailed information about the environment, they may also increase cost, weight, and deployment complexity \cite{gao2025}.

Smartphones are an attractive alternative because they combine a camera, inertial sensors, processing capabilities, and storage in a widely available device. Sensor data can provide supervision during dataset construction, whereas inference can rely exclusively on images. AIGD uses egocentric smartphone videos to predict local navigation directions and mobile sensor data to generate labels \cite{jadhav2025}. Similarly, \cite{medeiros2021} associated images, accelerometer data, and gyroscope data with forward and lateral movements performed during obstacle avoidance.

The construction and evaluation of such datasets still present important challenges. Consecutive frames often depict the same corridors, objects, and viewpoints; therefore, random frame-level splitting may distribute highly correlated samples across the training and test sets, causing information leakage and overly optimistic performance estimates \cite{figueiredo2024leakage}. In addition, natural walking sequences tend to overrepresent forward motion, whereas deviations and stops occur less frequently \cite{jadhav2025}. Diversity across buildings is also relevant, because differences in visual quality, data origin, or class composition may serve as visual shortcuts, without the model effectively learning the phenomenon of interest \cite{chitraranjan2025}.

This paper presents \sensecv{}, a multi-location egocentric visual--inertial dataset designed for local navigation in indoor environments. Data acquisition was supported by an open-source Android application for the synchronized recording of video, metadata, and sensor signals. The main contributions of this work are: (i) a dataset collected at three distinct indoor locations, containing controlled events of unobstructed forward motion, lateral deviation, and stopping; (ii) a semi-automatic labeling procedure that combines the collection protocol with accelerometer and gyroscope signals to temporally delimit the actions; (iii) a partitioning strategy based on recordings, scenes, and locations; and (iv) a multitask formulation that combines path condition and deviation direction to produce a local navigation action.

The experimental evaluation is guided by two hypotheses. The first investigates whether reducing the temporal sampling rate decreases redundancy among frames without substantially compromising model performance. The second examines whether splitting the data by recording, scene, and location produces more conservative estimates than a random frame-level split.

%##########################
% PT-BR

% A deficiência visual pode comprometer a autonomia em atividades de orientação e locomoção, especialmente em ambientes desconhecidos ou sujeitos a alterações. A Organização Mundial da Saúde estima que ao menos 2,2 bilhões de pessoas apresentam algum grau de deficiência visual para perto ou para longe \cite{who2026}. Em ambientes internos, obstáculos no piso ou suspensos, corredores estreitos, circulação de pessoas e mudanças na organização do espaço tornam o deslocamento mais complexo e podem aumentar a dependência de auxílio externo.

% Avanços em visão computacional e inteligência artificial têm permitido reconhecer caminhos transitáveis, detectar obstáculos, estimar a posição do usuário e produzir instruções de navegação \cite{casanova2025,khan2022}. Entretanto, muitas soluções dependem de câmeras RGB-D, LiDAR, sensores ultrassônicos, dispositivos vestíveis personalizados ou processamento externo. Embora esses recursos forneçam informações detalhadas, também podem elevar custo, peso e complexidade de implantação \cite{gao2025}.

% Smartphones constituem uma alternativa atraente por reunirem câmera, sensores inerciais, processamento e armazenamento em um dispositivo amplamente disponível. Os sensores podem fornecer supervisão durante a construção da base, enquanto a inferência final utiliza somente imagens. O AIGD emprega vídeos egocêntricos de smartphone para predizer direções locais e utiliza sensores móveis na geração dos rótulos \cite{jadhav2025}; de forma relacionada, \cite{medeiros2021} associaram imagens, acelerômetro e giroscópio a movimentos frontais e laterais em desvios de obstáculos.

% A construção e a avaliação dessas bases ainda apresentam desafios. Quadros consecutivos exibem os mesmos corredores, objetos e pontos de vista; por isso, a divisão aleatória pode distribuir amostras altamente correlacionadas entre treinamento e teste, provocando vazamento de informação e estimativas excessivamente otimistas \cite{figueiredo2024leakage}. Além disso, caminhadas naturais tendem a super-representar a continuidade frontal, enquanto desvios e paradas aparecem com menor frequência \cite{jadhav2025}. A diversidade entre edifícios também é relevante, pois diferenças de qualidade, origem ou composição das classes podem funcionar como atalhos visuais, sem que o fenômeno de interesse seja efetivamente aprendido \cite{chitraranjan2025}.

% Este artigo apresenta o \sensecv{}, uma base visual--inercial egocêntrica e multilocal voltada à navegação local em ambientes internos. A aquisição dos dados foi viabilizada por um aplicativo Android de código aberto para o registro sincronizado de vídeo, metadados e sinais sensoriais. As principais contribuições do trabalho são: (i) uma base coletada em três locais internos distintos, contendo eventos controlados de trajetória livre, desvio lateral e parada; (ii) um procedimento de rotulagem semiautomática que combina o protocolo de coleta com sinais do acelerômetro e do giroscópio para delimitar temporalmente as ações; (iii) uma estratégia de particionamento por gravação, cena e local; e (iv) uma formulação multitarefa que combina a condição da trajetória e a direção do desvio para produzir uma ação local de navegação.

% \textbf{A avaliação experimental é orientada por duas hipóteses. A primeira investiga se a redução da taxa de amostragem temporal diminui a redundância entre quadros sem comprometer substancialmente o desempenho dos modelos. A segunda verifica se a separação por gravação, cena e local produz estimativas mais conservadoras do que uma divisão aleatória por quadros.}

\section{Related Work}\label{sec:related_work}

Assistive visual navigation involves obstacle detection, traversable-area recognition, localization, and action prediction. Reviews in the field highlight the low cost and broad availability of camera-based solutions, while also identifying limitations related to illumination, occlusion, device placement, computational cost, and generalization across environments \cite{khan2022}. Among mobile solutions, \cite{breve2020} classified smartphone images as depicting either a free or non-free path, whereas Detectra uses 7,600 images and 76 obstacle classes in a YOLOv5s-based application \cite{bastidas2025detectra}. WOAD combines an infrared camera, depth sensing, FPGA-based processing, and multimodal feedback, providing richer perception at the cost of specialized hardware \cite{gao2025}.

Egocentric motion-prediction studies are more closely related to the present work. AIGD contains approximately 57 hours of video captured by a chest-mounted smartphone and predicts the classes \textsc{forward}, \textsc{left}, and \textsc{right}. Mobile sensors are used to generate the labels, while inference relies only on images; however, the dataset does not include a stopping class \cite{jadhav2025}. Similarly, \cite{medeiros2021} used images and inertial signals to estimate forward and lateral motion components, but adopted a frame-level split without explicit separation by recording, route, or environment.

Other approaches combine different sensors and architectures. \cite{plikynas2022} associated RGB images, depth data, and IMU signals with a ConvLSTM, but the system depends on wearable equipment, SLAM, cloud processing, and previously traversed routes. The multitask formulation also has precedents in DroNet, which jointly estimates steering direction and collision probability, and in HDIN, which discusses sensor noise and inconsistencies in visual-navigation labels \cite{loquercio2018,chang2022}.

Visual--inertial datasets have also been developed for other contexts. SCAND contains social-navigation demonstrations collected using Spot and Jackal robots \cite{karnan2022scand}, which can reach approximately 1.6 and 2.0~m/s, respectively. As a human reference, \cite{liu2019travel} considered an average walking speed of approximately 0.72~m/s for people with visual impairments. Differences in speed, camera height, and motion dynamics limit the direct transfer of robotic datasets to human-centered assistive navigation. The FLOBOT dataset also exhibited performance degradation across environments, reinforcing the importance of spatial diversity in evaluation \cite{yan2020flobot}.

The \sensecv{} dataset differs from these resources through its smartphone-based acquisition, inclusion of controlled deviation and stopping events, collection across multiple indoor environments, and partitioning strategy that prevents overlap between recordings and correlated scenes.

%##########################
% PT-BR

% A navegação visual assistiva abrange detecção de obstáculos, reconhecimento de regiões transitáveis, localização e predição de ações. Revisões da área destacam o baixo custo das câmeras, mas também limitações de iluminação, oclusão, posicionamento do dispositivo, custo computacional e generalização entre ambientes \cite{khan2022}.

% Soluções móveis tratam diferentes níveis do problema. \cite{breve2020} classificaram imagens de smartphone como caminho livre ou não livre, enquanto a Detectra reúne 7.600 imagens e 76 classes de obstáculos para uma aplicação baseada em YOLOv5s \cite{bastidas2025detectra}. Essas abordagens reconhecem transitabilidade ou objetos, mas não relacionam sequências egocêntricas à ação executada. O WOAD combina câmera infravermelha, profundidade, FPGA e retorno multimodal, oferecendo percepção mais rica ao custo de hardware especializado \cite{gao2025}.

% A predição egocêntrica de movimentos é mais próxima deste trabalho. O AIGD reúne aproximadamente 57 horas de vídeo capturado por smartphone no tórax e formula uma classificação multirrótulo de \textsc{frente}, \textsc{esquerda} e \textsc{direita}. Os sensores geram os rótulos, a inferência utiliza imagens e as cenas são separadas entre subconjuntos, mas não há classe de parada \cite{jadhav2025}. \cite{medeiros2021} discretizaram componentes frontal e lateral em cinco níveis de velocidade e avaliaram VGG16, ResNet-50 e DroNet. O trabalho demonstra o uso da IMU como supervisão visual, porém descreve a divisão no nível dos quadros, sem separação explícita por gravação, percurso ou ambiente.

% \cite{plikynas2022} associaram RGB, profundidade e IMU a uma ConvLSTM para predizer \textsc{frente}, \textsc{esquerda} e \textsc{direita}, mas o sistema depende de equipamento vestível, SLAM, nuvem e rotas previamente percorridas. A formulação multitarefa possui antecedentes em DroNet, que estima direção e probabilidade de colisão, e em HDIN, que discute ruído sensorial e inconsistências em rótulos de navegação visual de UAVs \cite{loquercio2018,chang2022}.

% Outras bases visual--inerciais possuem objetivos distintos. A AUT-VI reúne 126 sequências de vídeo, IMU e GPS em 17 locais para odometria visual--inercial \cite{samadzadeh2024}. A SCAND contém demonstrações sociais coletadas por robôs Spot e Jackal \cite{karnan2022scand}, capazes de atingir aproximadamente 1,6 e 2,0~m/s. Como referência humana, \cite{liu2019travel} consideraram velocidade média de caminhada de aproximadamente 0,72~m/s para pessoas com deficiência visual. Além da velocidade, altura da câmera, morfologia e sensores limitam a transferência direta de bases robóticas. A base FLOBOT, coletada por um robô de limpeza em múltiplos locais, também mostrou perdas entre ambientes, reforçando a importância da diversidade espacial \cite{yan2020flobot}. O \sensecv{} diferencia-se pela aquisição com smartphone, eventos controlados de desvio e parada, múltiplos ambientes internos e particionamento sem sobreposição de gravações e cenas.

\section{Methodology}\label{sec:proposed_system_methodology}

The methodology preserves the temporal organization of the data and keeps all samples derived from the same recording within a single subset, while exposing the models to different environments, objects, viewpoints, and lighting conditions. Figure~\ref{fig:collection_protocol} summarizes the collection events, while the following subsections describe data acquisition, the collection protocol, labeling, and the experimental evaluation.

\begin{figure}[t]
    \centering
    %\includegraphics[width=0.70\textwidth]{protocolo_coleta_small.png}
    \includegraphics[width=\columnwidth]{protocolo_coleta_small.png}
    \caption{Collection protocol: (A) left deviation, (B) right deviation, and (C) stopping. Deviation recordings begin about 5~m before the obstacle, with the maneuver starting at 1.5~m and ending after it is passed; stopping occurs approximately 1~m before the obstacle.}
    \label{fig:collection_protocol}
\end{figure}

%##########################
% PT-BR

% A metodologia foi estruturada para expor os modelos a diferentes edifícios, corredores, objetos, pontos de vista e condições de iluminação, preservando a organização temporal durante a rotulagem e o particionamento. Modelos de imagens podem utilizar quadros isolados e modelos temporais, clipes; em ambos os casos, todas as amostras derivadas da mesma gravação permanecem juntas. A Figura~\ref{fig:collection_protocol} resume os eventos com obstáculos, enquanto as subseções seguintes descrevem o aplicativo, o protocolo, a rotulagem e a avaliação.

% \begin{figure*}[t]
%     \centering
%     \includegraphics[width=0.70\textwidth]{protocolo_coleta_small.png}
%     \caption{Protocolo de coleta: (A) desvio à esquerda; (B) desvio à direita; e (C) parada. Nos desvios, a gravação começa cerca de 5~m antes do obstáculo, a manobra inicia a aproximadamente 1,5~m e termina após sua ultrapassagem. Na parada, o deslocamento é interrompido a cerca de 1~m do obstáculo.}
%     \label{fig:collection_protocol}
% \end{figure*}

\subsection{SenseCV Data Collection Application}

Data acquisition was performed using the open-source Android application \sensecv{}\footnote{Source code, installation instructions, and documentation: \url{https://github.com/SenseCV}.}. For each recording, the application stores the video, frame timestamps, device and capture configuration metadata, and JSON files containing accelerometer, gyroscope, and magnetometer signals. When location services are available, the application also records latitude and longitude coordinates. Each session is stored in a dedicated directory structure, facilitating data review, selection, and transfer.

The metadata include the device manufacturer and model, Android version, selected camera, focal length, orientation, resolution, frame rate, and other capture settings. The user can select the camera lens, portrait or landscape orientation, resolution, and a nominal frame rate of 15, 30, or 60~FPS. Recording can also be started and stopped using a Bluetooth button, reducing motion caused by interaction with the touchscreen. In the current version of this work, only accelerometer and gyroscope signals are used in the labeling procedure. Magnetometer and location data are preserved for future studies on trajectories, localization, and multimodal fusion.

%##########################
% PT-BR

% A aquisição foi realizada com o aplicativo Android de código aberto \sensecv{}\footnote{Código-fonte, instruções de instalação e documentação: \url{https://github.com/SenseCV}.}. Para cada gravação, o aplicativo armazena o vídeo, os timestamps dos quadros, os metadados do dispositivo e da configuração de captura e arquivos JSON contendo os sinais do acelerômetro, giroscópio e magnetômetro. Quando o serviço de localização está disponível, também são registradas as coordenadas de latitude e longitude, seus timestamps e o estado do GPS. Cada sessão é armazenada em uma estrutura própria de diretórios, facilitando a revisão, a seleção e a transferência dos dados.

% Os metadados incluem fabricante e modelo do dispositivo, versão do Android, câmera selecionada, distância focal, orientação, resolução, taxa de quadros e demais configurações de captura. O usuário pode selecionar a lente, a orientação em modo retrato ou paisagem, a resolução e uma taxa nominal de 15, 30 ou 60~FPS. A gravação também pode ser iniciada e interrompida por meio de um botão Bluetooth, reduzindo movimentos causados pela interação com a tela. Nesta versão do trabalho, somente os sinais do acelerômetro e do giroscópio são utilizados na rotulagem. Os dados do magnetômetro e da localização são preservados para estudos futuros de trajetórias, localização e fusão multimodal.

\subsection{Collection Protocol}

Data were collected between June and July 2026 in three indoor environments. Location A is an academic building with multiple blocks, corridors connecting classrooms and administrative areas, and varying lighting conditions; Location B is a library with bookshelf aisles, open circulation areas, and higher visual density; and Location C is another academic building with corridors of different widths, lighting variations, and frequent pedestrian traffic.

The dataset contains 592 valid videos distributed across 13 sessions, totaling 1~h, 45~min, and 42~s. Most recordings were captured with a Samsung Galaxy A56 using a 1.74~mm rear camera at a resolution of $480\times480$ pixels. One initial session was recorded with a POCO X3 Pro at $720\times720$ pixels.

Each video represents a complete navigation event: forward motion, left deviation, right deviation, or stopping. The conditions of each recording, including event type, obstacle position, corridor, smartphone orientation, and scene observations, were documented in a session spreadsheet. These records provide recording-level supervision and support verification of consistency among the planned action, the video, and the inertial signals. To reduce walking-induced oscillations and standardize acquisition, the smartphone was mounted on an S5 CamfreeGo gimbal approximately 1.30~m above the floor.

As shown in Figure~\ref{fig:collection_protocol}, deviation events began approximately 5~m before the obstacle, with the maneuver starting at about 1.5~m and continuing until the obstacle was passed. In some corridors at Location B, the initial distance was reduced to 3.2--3.7~m. During stopping events, the participant stopped approximately 1~m before the obstacle, whereas free-path events consisted of continuous forward walking, generally for ten or more steps.

Common circulation-area obstacles were used, including trash bins, cones, signs, boxes, chairs, fire extinguishers, stairs, books, and people in different positions. Repeated deviation, stopping, and free-path events were included to reduce the predominance of forward motion observed in datasets derived from natural walking sequences \cite{jadhav2025}.

%##########################
% PT-BR

% As coletas foram realizadas, entre junho e julho de 2026, em três ambientes internos distintos, denominados Local A, Local B e Local C. O Local A corresponde a um edifício acadêmico composto por diferentes blocos, com corredores de acesso a salas de aula e áreas administrativas, além de variações nas condições de iluminação. O Local B corresponde a uma biblioteca, com corredores entre estantes, áreas abertas de circulação, maior densidade de elementos visuais e iluminação predominantemente uniforme. O Local C corresponde a outro edifício acadêmico, também distribuído em diferentes blocos, com corredores de larguras variadas, mudanças de iluminação e circulação frequente de pessoas.

% A versão atual da base contém 592 vídeos válidos, distribuídos em 13 sessões de coleta e totalizando 1~h, 45~min e 42~s de duração. Na maior parte das sessões, foi utilizado um Samsung Galaxy A56, equipado com câmera traseira de distância focal de 1,74~mm e configurado com resolução de $480\times480$ pixels. Uma sessão inicial foi realizada com um POCO X3 Pro, utilizando resolução de $720\times720$ pixels.

% Cada vídeo corresponde a um evento completo de navegação, abrangendo uma das seguintes ações: seguir em frente, desviar para a esquerda, desviar para a direita ou parar. Antes de cada execução, as condições da coleta eram registradas em uma planilha específica da sessão, que associava um identificador à gravação e documentava o tipo de evento, a posição do obstáculo, o corredor utilizado, a orientação do smartphone e observações sobre a cena. Esses registros fornecem supervisão no nível da gravação e permitem confrontar a ação planejada com o movimento efetivamente observado no vídeo e nos sinais sensoriais.

% Testes preliminares indicaram que a fixação direta do smartphone ao corpo introduzia oscilações da caminhada. Por isso, o aparelho foi acoplado a um gimbal S5 CamfreeGo, posicionado a aproximadamente 1,30~m do piso, buscando reduzir movimentos involuntários e padronizar a altura. O estabilizador não elimina toda a variação, mas torna as condições de aquisição mais uniformes.

% Conforme ilustrado nos cenários A e B da Figura~\ref{fig:collection_protocol}, nos eventos de desvio, o responsável pela coleta (Person 1) iniciava a gravação aproximadamente 5~m antes do obstáculo, começava a manobra a cerca de 1,5~m dele e mantinha o deslocamento e a gravação por aproximadamente 1,5~m após ultrapassá-lo. O cenário A representa o desvio para a esquerda, enquanto o cenário B representa o desvio para a direita. Em alguns corredores do Local B, limitações físicas do ambiente exigiram distâncias iniciais menores, entre 3,2 e 3,7~m.

% Nos eventos de parada, representados no cenário C da Figura~\ref{fig:collection_protocol}, o Person 1 iniciava a aproximação a aproximadamente 5~m do obstáculo e interrompia o deslocamento cerca de 1~m antes dele. Os eventos de trajetória livre consistiam em caminhada contínua, sem obstáculo planejado interferindo no deslocamento, geralmente durante dez ou mais passos.

% Foram usados objetos comuns de circulação, incluindo lixeiras, cones, placas, caixas, cadeiras, extintores, escadas e livros, além de pessoas em diferentes posições e posturas. Repetições deliberadas de desvios, paradas e trajetórias livres reduzem a predominância frontal observada em bases obtidas de caminhadas naturais \cite{jadhav2025}.

\subsection{Labeling Scheme and Problem Formulation}

Each event is associated with two related but distinct supervised targets, defined in Equation~\ref{eq:targets}:

\begin{equation}
\begin{aligned}
c&\in\{\textsc{free},\textsc{blocked}\},\\
d&\in\{\textsc{left},\textsc{right},\textsc{no deviation}\}.
\end{aligned}
\label{eq:targets}
\end{equation}

Target $c$ represents the immediate path condition. A path is classified as \textsc{free} when the participant can continue moving forward without deviating or stopping, and as \textsc{blocked} when forward motion requires one of these actions. Therefore, the presence of an object in the image does not necessarily imply a blocked path if it does not interfere with the immediate trajectory.

Target $d$ represents the direction of the lateral maneuver. Deviation events are labeled as \textsc{left} or \textsc{right}, whereas forward motion and stopping events are labeled as \textsc{no deviation}. The combination of these two targets defines the local action $y\in\{\textsc{forward},\textsc{left},\textsc{right},\textsc{stop}\}$. The position of the obstacle does not directly determine the action because, for example, an object located on the right may require a deviation to the left.

The categorical label of each event is obtained from the action documented in the collection spreadsheet. The inertial signals do not replace this annotation; instead, they are used to temporally locate the motor response and verify its consistency with the recorded event. Thus, the procedure combines the supervision provided by the collection protocol with the automatic analysis of accelerometer and gyroscope signals.

Video and IMU data therefore play complementary roles. Inertial signals are used to estimate $T_1$ and $T_2$, corresponding respectively to the beginning and end of the motor response, whereas the video provides the visual context of the event. Because the decision should precede the execution of the action, the visual input $X_t$ is selected before $T_1$, reducing the risk of including frames that already reveal the performed maneuver.

The procedure uses accelerometer and gyroscope sequences synchronized with the video timestamps. First, the gravity direction is estimated using a centered moving average. This estimate defines a local coordinate system composed of a vertical axis, a forward axis approximately aligned with the projection of the camera optical axis, and a lateral axis \cite{manos2019gravity}.

Two descriptors are calculated for deviation detection: the angular velocity around the vertical axis, obtained from the gyroscope, and an estimate of lateral velocity, obtained by integrating acceleration along the lateral axis. Before integration, the low-frequency component of the acceleration is removed to limit accumulated drift \cite{munozdiaz2017drift}. Because the gimbal may attenuate rotations of the device, lateral velocity is used as the reference descriptor when the magnitude of the angular velocity does not remain above its threshold for the minimum persistence interval.

A deviation is detected when the magnitude of the selected descriptor exceeds its threshold and remains above it for the required minimum interval. The first instant of this persistence interval defines $T_1$, whereas $T_2$ corresponds to the instant at which the descriptor returns below the threshold. The dominant sign between these instants, interpreted according to the recorded smartphone orientation, determines whether the movement occurred to the left or to the right.

Stopping events are identified through a sustained reduction in inertial activity following a walking period. The local variability of the acceleration magnitude is used to distinguish walking, deceleration, and low inertial activity, while a complementary detector verifies the loss of step periodicity. This procedure is consistent with zero-velocity detection and gait-cycle analysis methods \cite{park2010zerovelocity,moenilssen2004gait}.

Because gait parameters may differ between people with and without visual impairments \cite{rasouli2023gait}, the thresholds were calibrated for the devices, gimbal, and collection protocol used in this work. The adopted parameters were a decision lead time of $\Delta=1.0$~s, a minimum persistence of 0.4~s for deviation validation, a posterior margin of $m=0.3$~s, an angular-velocity threshold of $25^\circ$/s, a lateral-velocity threshold of 0.15~m/s, an acceleration variability threshold of 0.025~m/s\textsuperscript{2}, and a minimum duration of 0.5~s for stopping detection. The angular and lateral thresholds are used to identify the maneuver, whereas acceleration variability and minimum duration prevent brief reductions in inertial activity from being interpreted as stops.

From instants $T_1$ and $T_2$, the three temporal windows in Equation~\ref{eq:windows} are defined:

{
\setlength{\abovedisplayskip}{4pt}
\setlength{\belowdisplayskip}{4pt}
\setlength{\abovedisplayshortskip}{2pt}
\setlength{\belowdisplayshortskip}{2pt}
\begin{equation}
\begin{aligned}
W_{\text{action}}&=[T_1,T_2],\\
W_{\text{decision}}&=[T_1-\Delta,T_1),\\
W_{\text{expanded}}&=[T_1-\Delta,T_2+m].
\end{aligned}
\label{eq:windows}
\end{equation}
}

The window $W_{\text{action}}$ delimits the execution of the motor response. $W_{\text{decision}}$ contains the interval preceding the beginning of the action and is used to select the images or clips provided to the model. $W_{\text{expanded}}$ includes the preceding context, the execution of the action, and a posterior margin, and is used to generate clips for inspection.

The detector searches for an interval consistent with the spreadsheet annotation. For deviation events, the identified direction must agree with the documented direction; for stopping events, a sustained transition to low inertial activity must be observed; and for forward motion events, neither a deviation nor a stop should be detected. Inconsistent events receive a low-confidence designation and are forwarded for review without automatically changing the original label.

The clips corresponding to $W_{\text{expanded}}$ are organized into a review video containing the documented action and the estimated temporal boundaries. A quality report is also produced, considering the effective sampling rate, timestamp gaps, possible sensor saturation, and preservation of step periodicity. An event is discarded only when neither the video nor the inertial signals allow the action or its temporal boundaries to be reliably confirmed.

The model uses a shared feature extractor and produces $f_\theta(X_t)=(\hat c_t,\hat d_t)$. The two predictions are combined deterministically according to Equation~\ref{eq:action}:

{
\setlength{\abovedisplayskip}{4pt}
\setlength{\belowdisplayskip}{4pt}
\setlength{\abovedisplayshortskip}{2pt}
\setlength{\belowdisplayshortskip}{2pt}
\begin{equation}
\hat y_t=\begin{cases}
\textsc{left},&\hat d_t=\textsc{left},\\
\textsc{right},&\hat d_t=\textsc{right},\\
\textsc{forward},&
\hat d_t=\textsc{no deviation}\!\land\!\hat c_t=\textsc{free},\\
\textsc{stop},&
\hat d_t=\textsc{no deviation}\!\land\!\hat c_t=\textsc{blocked}.
\end{cases}
\label{eq:action}
\end{equation}
}

In Equation~\ref{eq:action}, when the deviation head predicts \textsc{left} or \textsc{right}, the predicted direction directly determines the final action. When the prediction is \textsc{no deviation}, the path condition head distinguishes forward motion from stopping. The \textsc{stop} label represents the local response observed during the collection protocol and does not imply the absence of alternative routes outside the camera field of view.

%##########################
% PT-BR

% Cada evento é associado a dois alvos supervisionados relacionados, porém distintos, definidos na Equação~\ref{eq:targets}:

% \begin{equation}
% \begin{aligned}
% c&\in\{\textsc{livre},\textsc{bloqueada}\},\\
% d&\in\{\textsc{esquerda},\textsc{direita},\textsc{sem desvio}\}.
% \end{aligned}
% \label{eq:targets}
% \end{equation}

% O alvo $c$ representa a condição da trajetória imediata. A trajetória é classificada como \textsc{livre} quando o participante pode continuar o deslocamento frontal sem realizar um desvio ou uma parada, e como \textsc{bloqueada} quando a continuidade exige uma dessas ações. Portanto, a presença de um objeto na imagem não implica necessariamente bloqueio, caso ele não interfira no deslocamento imediato.

% O alvo $d$ representa a direção da manobra lateral. Eventos de desvio recebem os rótulos \textsc{esquerda} ou \textsc{direita}, enquanto eventos de continuidade frontal e de parada recebem \textsc{sem desvio}. A combinação dos dois alvos define a ação local $y\in\{\textsc{frente},\textsc{esquerda},\textsc{direita},\textsc{parar}\}$. A posição do obstáculo não determina diretamente essa ação, pois, por exemplo, um objeto localizado à direita pode resultar em um desvio para a esquerda.

% O rótulo categórico de cada evento é obtido a partir da ação documentada na planilha de coleta. Os sinais inerciais não substituem essa anotação, mas são utilizados para localizar temporalmente a resposta motora e verificar sua compatibilidade com o evento registrado. Dessa forma, o procedimento combina a supervisão fornecida pelo protocolo com a análise automática dos sinais do acelerômetro e do giroscópio.

% Vídeo e IMU desempenham, portanto, papéis complementares. Os sinais inerciais são utilizados para estimar $T_1$ e $T_2$, correspondentes, respectivamente, ao início e ao término da resposta motora, enquanto o vídeo fornece o contexto visual do evento. Como a decisão deve anteceder a execução da ação, a entrada visual $X_t$ é selecionada antes de $T_1$, reduzindo o risco de incluir quadros que já revelem a manobra executada.

% O procedimento utiliza as sequências do acelerômetro e do giroscópio sincronizadas com os timestamps do vídeo. Inicialmente, a direção da gravidade é estimada por meio de uma média móvel centrada. A partir dela, define se uma base local formada por um eixo vertical, um eixo frontal aproximadamente alinhado à projeção do eixo óptico da câmera e um eixo lateral \cite{manos2019gravity}.

% Para a detecção dos desvios, são calculados dois descritores: a velocidade angular em torno do eixo vertical, obtida do giroscópio, e uma estimativa da velocidade lateral, calculada pela integração da aceleração no eixo lateral. Antes da integração, a componente de baixa frequência da aceleração é removida para limitar a deriva acumulada \cite{munozdiaz2017drift}. Como o gimbal pode atenuar as rotações do aparelho, a velocidade lateral é utilizada como descritor de referência quando o módulo da velocidade angular não permanece acima do limiar durante o intervalo mínimo de persistência.

% Um desvio é detectado quando o módulo do descritor selecionado ultrapassa seu limiar e permanece nessa condição durante o intervalo mínimo estabelecido. O primeiro instante dessa persistência define $T_1$, enquanto $T_2$ corresponde ao retorno do descritor para abaixo do limiar. A polaridade predominante entre esses instantes, interpretada segundo a orientação registrada do smartphone, determina se o movimento ocorreu para a esquerda ou para a direita.

% Os eventos de parada são identificados por uma redução sustentada da dinâmica inercial após um período de caminhada. A variabilidade local da magnitude da aceleração é utilizada para distinguir caminhada, desaceleração e baixa dinâmica, enquanto um detector complementar verifica a perda da periodicidade das passadas. Esse procedimento é compatível com métodos de detecção de velocidade zero e análise do ciclo da marcha \cite{park2010zerovelocity,moenilssen2004gait}.

% Como parâmetros de marcha podem variar entre pessoas com e sem deficiência visual \cite{rasouli2023gait}, os limiares foram calibrados para os dispositivos, o gimbal e o protocolo empregados na coleta. Foram adotados uma antecipação da decisão de $\Delta=1{,}0$~s, persistência mínima de 0,4~s para validação dos desvios, margem posterior de $m=0{,}3$~s, limiar de velocidade angular de $25^\circ$/s, limiar de velocidade lateral de 0,15~m/s, limiar de variabilidade da aceleração de 0,025~m/s\textsuperscript{2} e duração mínima de 0,5~s para a detecção de paradas. Os limiares angular e lateral são utilizados para identificar a manobra, enquanto a variabilidade da aceleração e a duração mínima evitam que reduções momentâneas da dinâmica sejam interpretadas como paradas.

% A partir dos instantes $T_1$ e $T_2$, são definidas as três janelas temporais da Equação~\ref{eq:windows}:

% {
% \setlength{\abovedisplayskip}{4pt}
% \setlength{\belowdisplayskip}{4pt}
% \setlength{\abovedisplayshortskip}{2pt}
% \setlength{\belowdisplayshortskip}{2pt}
% \begin{equation}
% \begin{aligned}
% W_{\text{ação}}&=[T_1,T_2],\\
% W_{\text{decisão}}&=[T_1-\Delta,T_1),\\
% W_{\text{expandida}}&=[T_1-\Delta,T_2+m].
% \end{aligned}
% \label{eq:windows}
% \end{equation}
% }

% A janela $W_{\text{ação}}$ delimita a execução da resposta motora. $W_{\text{decisão}}$ contém o intervalo anterior ao início da ação e é utilizada para selecionar as imagens ou os clipes fornecidos ao modelo. $W_{\text{expandida}}$ inclui o contexto anterior, a execução da ação e uma margem posterior, sendo utilizada para gerar os cortes destinados à inspeção.

% O detector procura um intervalo compatível com a anotação da planilha. Nos eventos de desvio, o lado identificado deve concordar com a direção documentada; nas paradas, deve ser observada uma transição sustentada para baixa dinâmica; e, nos eventos frontais, não deve ser detectado desvio ou parada. Divergências recebem baixa confiança e são encaminhadas para revisão, sem alteração automática do rótulo original.

% Os cortes correspondentes a $W_{\text{expandida}}$ são organizados em um vídeo de revisão contendo a ação documentada e os limites temporais estimados. Também é produzido um relatório de qualidade que considera a taxa efetiva de amostragem, lacunas nos timestamps, possíveis saturações dos sensores e a preservação da periodicidade das passadas. Um evento somente é descartado quando o vídeo e os sinais inerciais não permitem confirmar a ação ou seus limites temporais.

% O modelo utiliza um extrator de características compartilhado e produz $f_\theta(X_t)=(\hat c_t,\hat d_t)$. As duas predições são combinadas de forma determinística conforme a Equação~\ref{eq:action}:

% \begin{equation}
% \setlength{\abovedisplayskip}{4pt}
% \setlength{\belowdisplayskip}{4pt}
% \setlength{\abovedisplayshortskip}{2pt}
% \setlength{\belowdisplayshortskip}{2pt}
% \hat y_t=\begin{cases}
% \textsc{esquerda},&\hat d_t=\textsc{esquerda},\\
% \textsc{direita},&\hat d_t=\textsc{direita},\\
% \textsc{frente},&
% \hat d_t=\textsc{sem desvio}\!\land\!\hat c_t=\textsc{livre},\\
% \textsc{parar},&
% \hat d_t=\textsc{sem desvio}\!\land\!\hat c_t=\textsc{bloqueada}.
% \end{cases}
% \label{eq:action}
% \end{equation}

% Na Equação~\ref{eq:action}, quando a cabeça de desvio prediz \textsc{esquerda} ou \textsc{direita}, essa direção determina diretamente a ação final. Quando a predição é \textsc{sem desvio}, a cabeça de trajetória diferencia a continuidade frontal da parada. O rótulo \textsc{parar} representa a resposta local observada durante o protocolo e não implica a ausência de rotas alternativas fora do campo de visão da câmera.

\subsection{Dataset Statistics and Partitioning Strategy}

The current version of the dataset contains 592 valid events, of which 320 were collected at Location A, 95 at Location B, and 177 at Location C. For the path condition head, 142 events belong to the \textsc{free} class and 450 to the \textsc{blocked} class. For the deviation head, 197 events are labeled as \textsc{left}, 187 as \textsc{right}, and 208 as \textsc{no deviation}. The deterministic combination of these two targets results in 142 \textsc{forward}, 197 \textsc{left}, 187 \textsc{right}, and 66 \textsc{stop} actions.

Partitioning is performed at the recording level, before frame extraction or clip generation. Each video and all samples derived from it remain entirely within a single subset. In addition, recordings captured in the same corridor or under similar scene configurations are kept together. This strategy prevents consecutive frames, overlapping clips, or nearly identical images from being distributed across the training, validation, and test sets.

In the adopted inter-location protocol, Location C was reserved exclusively for testing, whereas recordings from Locations A and B were used to compose the training and validation sets.

Table~\ref{tab:split_distribution} presents the total number of samples used in each subset with 30 FPS and their distribution across the classes of the two classification heads. Because each sample simultaneously has a path condition label and a deviation label, the totals for both heads correspond to the same set of images.


\begin{table}[t]
\caption{Distribution of samples across subsets and classes of the two
classification heads.}
\label{tab:split_distribution}
\centering
\scriptsize
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{2.5pt}

\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Subset} &
\textbf{Total} &
\multicolumn{2}{c}{\textbf{Path Status}} &
\multicolumn{3}{c}{\textbf{Deviation}}\\
\cmidrule(lr){3-4}
\cmidrule(lr){5-7}
&
&
\textbf{Free} &
\textbf{Blocked} &
\textbf{Left} &
\textbf{Right} &
\shortstack{\textbf{No}\\\textbf{deviation}}\\
\midrule

Training
& 33\,188
& 16\,105
& 17\,083
& 7\,094
& 7\,222
& 18\,872\\

Validation
& 6\,102
& 2\,953
& 3\,149
& 1\,214
& 1\,284
& 3\,604\\

Test
& 3\,970
& 1\,740
& 2\,230
& 1\,247
& 983
& 1\,740\\

\bottomrule
\end{tabular}%
}
\end{table}

The recording- and scene-level separation was adopted to reduce the risk of information leakage in a dataset derived from videos. In \cite{medeiros2021}, the images were reorganized and divided without an explicit grouping by recording or scene. Under this type of partitioning, temporally adjacent or visually correlated frames may be distributed across different subsets, producing overly optimistic performance estimates \cite{figueiredo2024leakage}.

In \sensecv{}, all samples derived from the same recording remain within the same subset, and testing is performed exclusively using data from Location C. Therefore, the inter-location evaluation aims to measure the ability of the models to generalize to corridors, objects, lighting conditions, and visual configurations not observed during training.

The raw data corresponding to the test set are publicly available on Kaggle\footnote{Recordings, sensor signals, and metadata from the test set: \url{https://www.kaggle.com/datasets/sensecv/test-set}.}, preserving the original session organization. The training and validation data will be released at a later stage.

%##########################
% PT-BR

% A versão atual da base reúne 592 eventos válidos, dos quais 320 foram coletados no Local A, 95 no Local B e 177 no Local C. Na cabeça de trajetória, 142 eventos pertencem à classe \textsc{livre} e 450 à classe \textsc{bloqueada}. Na cabeça de desvio, 197 eventos são classificados como \textsc{esquerda}, 187 como \textsc{direita} e 208 como \textsc{sem desvio}. A combinação determinística desses dois alvos resulta em 142 ações \textsc{frente}, 197 \textsc{esquerda}, 187 \textsc{direita} e 66 \textsc{parar}. 

% O particionamento é realizado no nível das gravações, antes da extração dos quadros ou da formação dos clipes. Cada vídeo e todas as amostras dele derivadas permanecem integralmente em um único subconjunto. Além disso, gravações realizadas no mesmo corredor ou sob uma configuração de cena semelhante são mantidas juntas. Essa estratégia evita que quadros consecutivos, clipes sobrepostos ou imagens visualmente quase idênticas sejam distribuídos entre treinamento, validação e teste.

% No protocolo interlocal adotado, o Local C foi reservado exclusivamente para o conjunto de teste, enquanto as gravações dos Locais A e B foram utilizadas na composição dos conjuntos de treinamento e validação.


% A Tabela~\ref{tab:split_distribution} apresenta a quantidade de amostras utilizadas em cada subconjunto e sua distribuição entre as classes das duas cabeças de classificação. Como cada amostra possui simultaneamente um rótulo de trajetória e um rótulo de desvio, os totais de cada cabeça correspondem ao mesmo conjunto de imagens.

% \textbf{[Descrever a política de amostragem temporal utilizada para gerar as amostras: número de quadros extraídos por evento, janela temporal considerada, taxa de amostragem e tratamento de vídeos com diferentes taxas de quadros. Essa informação é necessária para reproduzir os totais de 33\,188, 6\,102 e 3\,970 amostras.]}

% \begin{table}[t]
% \caption{Distribuição das amostras entre os subconjuntos e as classes das duas cabeças de classificação.}
% \label{tab:split_distribution}
% \centering
% \scriptsize
% \renewcommand{\arraystretch}{1.0}
% \setlength{\tabcolsep}{3.5pt}
% \begin{tabular}{lllr}
% \toprule
% \textbf{Subconjunto} &
% \textbf{Cabeça} &
% \textbf{Classe} &
% \textbf{Amostras} \\
% \midrule
% Treinamento
%     & Trajetória & Livre      & 16\,105 \\
%     &            & Bloqueada  & 17\,083 \\
%     & Desvio     & Esquerda   & 7\,094 \\
%     &            & Direita    & 7\,222 \\
%     &            & Sem desvio & 18\,872 \\
%     & \multicolumn{2}{l}{\textbf{Total}}
%     & \textbf{33\,188} \\
% \midrule
% Validação
%     & Trajetória & Livre      & 2\,953 \\
%     &            & Bloqueada  & 3\,149 \\
%     & Desvio     & Esquerda   & 1\,214 \\
%     &            & Direita    & 1\,284 \\
%     &            & Sem desvio & 3\,604 \\
%     & \multicolumn{2}{l}{\textbf{Total}}
%     & \textbf{6\,102} \\
% \midrule
% Teste
%     & Trajetória & Livre      & 1\,740 \\
%     &            & Bloqueada  & 2\,230 \\
%     & Desvio     & Esquerda   & 1\,247 \\
%     &            & Direita    & 983 \\
%     &            & Sem desvio & 1\,740 \\
%     & \multicolumn{2}{l}{\textbf{Total}}
%     & \textbf{3\,970} \\
% \bottomrule
% \end{tabular}
% \end{table}

% \textbf{[Confirmar a distribuição das quatro ações finais em cada subconjunto. Os valores atuais sugerem que o conjunto de teste pode não conter amostras da classe \textsc{parar}, pois todas as amostras \textsc{bloqueada} parecem estar associadas a desvios para a esquerda ou para a direita. Essa condição afeta o cálculo da macro-F1 da ação final.]}

% A separação por gravação e por cena foi adotada para reduzir o risco de vazamento de informação em uma base derivada de vídeos. Em \cite{medeiros2021}, as imagens foram reorganizadas e divididas sem que fosse descrito um agrupamento explícito por gravação ou cena. Nesse tipo de particionamento, quadros temporalmente próximos ou visualmente correlacionados podem aparecer em subconjuntos distintos, produzindo estimativas excessivamente otimistas \cite{figueiredo2024leakage}.

% No \sensecv{}, todas as amostras derivadas de uma mesma gravação permanecem no mesmo subconjunto, e o teste é realizado exclusivamente com dados do Local C. Assim, a avaliação interlocal busca medir a capacidade de generalização dos modelos para corredores, objetos, condições de iluminação e configurações visuais não observados durante o treinamento.

% \textbf{[Após a realização do experimento de controle, inserir nesta subseção ou na Configuração Experimental a comparação entre divisão aleatória por quadros, separação por gravação/cena e separação interlocal. Esse experimento é necessário para validar quantitativamente a discussão sobre vazamento de dados.]}

% Os dados brutos correspondentes ao conjunto de teste estão disponíveis no Kaggle\footnote{Gravações, sinais sensoriais e metadados do conjunto de teste: \url{https://www.kaggle.com/datasets/sensecv/test-set}.}, preservando a organização original das sessões. Os dados dos conjuntos de treinamento e validação serão disponibilizados posteriormente.

\subsection{Experimental Setup}

The MobileNetV1 \cite{howard2017mobilenets}, EfficientNet-B0 \cite{tan2019efficientnet}, ResNet-50 \cite{he2016resnet}, and MobileNetV4-Conv-Small \cite{qin2024mobilenetv4} models were evaluated. The feature extractors were initialized with ImageNet-pretrained weights and extended with two linear output layers: a binary head for classifying the path condition as \textsc{free} or \textsc{blocked}, and a three-class head for distinguishing among \textsc{left}, \textsc{right}, and \textsc{no deviation}.

The images were resized to $224\times224$ pixels using bicubic interpolation and normalized using the ImageNet means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225). Training was performed with batches of 32 images using the Adam optimizer, with an initial learning rate of $10^{-4}$ and a weight decay of $10^{-4}$. The objective function was defined as the equally weighted sum of the cross-entropy losses from the two output heads.

Training was limited to 40 epochs, with early stopping after six consecutive epochs without a reduction in the validation loss. For the final evaluation, the model state corresponding to the lowest validation loss was restored.

Horizontal flipping was applied to every image during training as a data augmentation technique. Under this transformation, the \textsc{left} and \textsc{right} labels were exchanged, whereas the \textsc{no deviation} label and the path condition labels remained unchanged.

The training, validation, and test partitions, as well as the preprocessing, data augmentation, optimization, and model-selection procedures, were kept fixed across architectures. Therefore, all models were compared under a common experimental configuration.

To quantify, as a baseline reference only, the extent to which data leakage inflates apparent performance --- and not to be interpreted as a valid estimate of real-world accuracy --- two additional splitting strategies were tested:

\begin{itemize}
    \item \textbf{Recording-level split (spatial leakage):} Entire videos are separated between training and test sets, but all recordings come from the same institution.
    \item \textbf{Random frame split (temporal + spatial leakage):} Frames from the same video are mixed across the training and test sets.
\end{itemize}

These leakage-baseline experiments were conducted only with MobileNetV1 and MobileNetV4-Conv-Small, chosen for having shown the best performance in earlier experiments. Each configuration was run 5 times, once using all frames of the dataset and once subsampling 1 in every 3 frames (10 FPS). These results serve solely to illustrate the accuracy gains attributable to leakage and should not be taken as representative of true model generalization.
%##########################
% PT-BR

% Foram avaliados os modelos MobileNetV1 \cite{howard2017mobilenets}, EfficientNet-B0 \cite{tan2019efficientnet}, ResNet-50 \cite{he2016resnet} e MobileNetV4-Conv-Small \cite{qin2024mobilenetv4}. Os extratores de características foram inicializados com pesos pré-treinados no ImageNet e receberam duas camadas lineares de saída: uma cabeça binária para classificar a trajetória como \textsc{livre} ou \textsc{bloqueada} e uma cabeça de três classes para distinguir \textsc{esquerda}, \textsc{direita} e \textsc{sem desvio}.

% \textbf{[Informar se todos os parâmetros dos extratores foram ajustados durante o treinamento ou se alguma parte das redes permaneceu congelada.]}

% As imagens foram redimensionadas para $224\times224$ pixels por interpolação bicúbica e normalizadas utilizando as médias $(0{,}485,0{,}456,0{,}406)$ e os desvios padrão $(0{,}229,0{,}224,0{,}225)$ do ImageNet. O treinamento utilizou lotes de 32 \textbf{imagens?} e o otimizador Adam, com taxa de aprendizado inicial de $10^{-4}$ e decaimento de pesos de $10^{-4}$. A função objetivo correspondeu à soma, com pesos iguais, das perdas de entropia cruzada das duas cabeças.

% O treinamento foi limitado a 40 épocas, com parada antecipada após seis épocas consecutivas sem redução da perda de validação. Para a avaliação final, foi recuperado o estado do modelo correspondente à menor perda observada no conjunto de validação.

% Durante o treinamento, foi aplicado espelhamento
% horizontal como técnica de aumento de dados. Nessa transformação, os rótulos \textsc{esquerda} e \textsc{direita} foram trocados entre si, enquanto \textsc{sem desvio} e os rótulos da cabeça de trajetória foram preservados.

% \textbf{[Informar a probabilidade de aplicação do espelhamento horizontal, caso ele não tenha sido aplicado a todas as imagens.]}

% As divisões de treinamento, validação e teste, assim como os procedimentos de pré-processamento, aumento de dados, otimização e seleção do melhor estado, foram mantidos fixos entre as arquiteturas. Dessa forma, os modelos foram comparados sob uma configuração experimental comum.

% Os experimentos foram executados uma única vez, utilizando uma semente aleatória fixa. A implementação foi realizada com \textbf{[framework e versão]}, e o treinamento foi conduzido em \textbf{[hardware utilizado]}.

% \textbf{[Informar o valor da semente aleatória.]}

\section{Results and Discussion}\label{sec:results_discussions}

All experiments were repeated five times, and the results are reported using the mean and standard deviation across runs. Table~\ref{tab:results_location_split} presents the performance on the test set composed exclusively of data from Location C, whose environments were not observed during training.
\begin{table*}[t]
    \caption{Performance of the evaluated models on the Location C test set and final action evaluation results over five runs.}
    \label{tab:results_location_split}
    \centering
    \scriptsize
    \renewcommand{\arraystretch}{1.00}
    \setlength{\tabcolsep}{3pt}

    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lrrrrrrr}
    \toprule
    \textbf{Model} &
    \textbf{Parameters} &
    \multicolumn{2}{c}{\textbf{Path Status (\%)}} &
    \multicolumn{2}{c}{\textbf{Deviation (\%)}} &
    \multicolumn{2}{c}{\textbf{Final Action (\%)}}\\
    \cmidrule(lr){3-4}
    \cmidrule(lr){5-6}
    \cmidrule(lr){7-8}
    &
    &
    \textbf{Accuracy} &
    \textbf{Macro-F1} &
    \textbf{Accuracy} &
    \textbf{Macro-F1} &
    \textbf{Accuracy} &
    \textbf{Macro-F1}\\
    \midrule

    MobileNetV1
    & 3.21 M
    & 91.86 $\pm$ 3.40
    & 91.84 $\pm$ 3.38
    & 84.67 $\pm$ 7.14
    & 84.29 $\pm$ 7.41
    & 84.42 $\pm$ 8.16
    & 86.37 $\pm$ 7.23\\

    EfficientNet-B0
    & 4.01 M
    & 92.09 $\pm$ 1.05
    & 92.07 $\pm$ 1.04
    & 79.93 $\pm$ 5.31
    & 77.93 $\pm$ 6.15
    & 79.67 $\pm$ 6.40
    & 80.13 $\pm$ 4.89\\

    ResNet-50
    & 23.52 M
    & 91.20 $\pm$ 1.01
    & 91.14 $\pm$ 1.03
    & 81.42 $\pm$ 12.02
    & 79.61 $\pm$ 15.01
    & 79.36 $\pm$ 15.03
    & 81.89 $\pm$ 14.62\\

    MobileNetV4
    & \textbf{2.50 M}
    & \textbf{95.58 $\pm$ 0.63}
    & \textbf{95.54 $\pm$ 0.64}
    & \textbf{89.97 $\pm$ 6.42}
    & \textbf{89.80 $\pm$ 6.74}
    & \textbf{89.85 $\pm$ 7.56}
    & \textbf{91.39 $\pm$ 5.25}\\

    \bottomrule
    \end{tabular}%
    }
\end{table*}

MobileNetV4 achieved the best performance on both tasks, reaching 95.58\% accuracy and 95.54\% macro-F1 for path status, and 89.97\% and 89.80\%, respectively, for deviation direction. In addition to outperforming the other architectures, it has the smallest number of parameters, indicating the best trade-off between predictive performance and computational cost. MobileNetV1 obtained the second-best result for deviation classification, while EfficientNet-B0 showed competitive performance for path status but lower results for deviation. ResNet-50, despite being the largest architecture, provided no advantage and exhibited the highest variability across runs.

The deterministic combination of the two heads produced comparable final-action results. MobileNetV4 achieved 89.85\% accuracy and 91.39\% macro-F1 for the four local actions, showing that the separate path-status and deviation predictions can be combined effectively into the navigation response defined in Equation~\ref{eq:action}. The lower final-action accuracy relative to path status is consistent with the greater difficulty and variability of deviation classification.

Across all architectures, path-status classification was more stable than deviation classification. The macro-F1 values close to accuracy for the first task indicate relatively balanced performance between the \textsc{free} and \textsc{blocked} classes. In contrast, the larger standard deviations observed for the second task suggest greater sensitivity to initialization, batch composition, and the visual characteristics of the unseen environment.

Table~\ref{tab:configuration_results} highlights the effect of the partitioning strategy. Random frame-level splitting produced the highest results, reaching approximately 100\% for path status and more than 96\% accuracy for deviation classification. However, under this configuration, temporally adjacent frames from the same recording may be distributed across the training and test sets. Therefore, these values represent a reference for the effect of information leakage rather than a valid estimate of generalization.


\begin{table}[t]
  \caption{Performance of the evaluated models under different temporal sampling rates and data partitioning strategies.}
  \label{tab:configuration_results}
  \centering
  \scriptsize
  \renewcommand{\arraystretch}{1.08}
  \setlength{\tabcolsep}{1.7pt}
  \resizebox{\columnwidth}{!}{%
  \begin{tabular}{llrrrr}
  \toprule
  \textbf{Model} &
  \textbf{Configuration} &
  \multicolumn{2}{c}{\textbf{Path Status (\%)}} &
  \multicolumn{2}{c}{\textbf{Deviation (\%)}}\\
  \cmidrule(lr){3-4}
  \cmidrule(lr){5-6}
  &
  &
  \textbf{Accuracy} &
  \textbf{Macro-F1} &
  \textbf{Accuracy} &
  \textbf{Macro-F1}\\
  \midrule
  MobileNetV1
  & 30 FPS + Random Frames
  & 99.97 $\pm$ 0.02
  & 99.97 $\pm$ 0.02
  & 96.11 $\pm$ 0.83
  & 92.82 $\pm$ 1.35\\
  & 30 FPS + Random Clips
  & \textbf{98.87 $\pm$ 0.51}
  & \textbf{98.78 $\pm$ 0.50}
  & 91.82 $\pm$ 2.11
  & 84.62 $\pm$ 4.27\\
  & 10 FPS + Random Frames
  & 99.71 $\pm$ 0.25
  & 99.63 $\pm$ 0.33
  & 94.83 $\pm$ 0.85
  & 88.80 $\pm$ 1.92\\
  & 10 FPS + Random Clips
  & 98.78 $\pm$ 0.82
  & 98.31 $\pm$ 1.10
  & \textbf{94.41 $\pm$ 1.67}
  & \textbf{86.83 $\pm$ 4.08}\\
  \midrule
  MobileNetV4
  & 30 FPS + Random Frames
  & 99.99 $\pm$ 0.02
  & 99.99 $\pm$ 0.02
  & 96.47 $\pm$ 0.41
  & 93.40 $\pm$ 0.76\\
  & 30 FPS + Random Clips
  & 98.81 $\pm$ 0.46
  & \textbf{98.71 $\pm$ 0.46}
  & 91.43 $\pm$ 2.45
  & 84.46 $\pm$ 4.21\\
  & 10 FPS + Random Frames
  & 99.75 $\pm$ 0.17
  & 99.68 $\pm$ 0.22
  & 95.24 $\pm$ 0.74
  & 89.64 $\pm$ 1.53\\
  & 10 FPS + Random Clips
  & \textbf{98.88 $\pm$ 0.49}
  & 98.45 $\pm$ 0.65
  & \textbf{94.59 $\pm$ 1.82}
  & \textbf{87.42 $\pm$ 3.41}\\
  \bottomrule
  \end{tabular}%
  }
\end{table}

When temporal grouping was preserved through clip-level splitting, performance decreased, particularly for macro-F1 in deviation classification. This difference indicates that distributing temporally correlated frames across the training and test sets inflates the measured performance. The result is consistent with the concerns regarding frame-level partitioning discussed in \cite{medeiros2021}. In contrast, the protocol adopted in \sensecv{} keeps recordings and correlated scenes within the same subset and uses an exclusive location for testing, resulting in a more conservative evaluation.

Reducing the temporal sampling rate from 30 to 10~FPS had little effect on path-status classification. For both architectures, the difference remained below 0.5 percentage points across the evaluated configurations. Under random frame-level splitting, a small decrease was observed for deviation classification. However, under clip-level splitting, 10~FPS produced better results than 30~FPS for this task. These findings indicate that reducing the sampling rate removes part of the temporal redundancy without substantially compromising performance and may benefit learning when frame correlation is controlled.

Although AIGD also preserves scene separation \cite{jadhav2025}, differences in classes, collection protocol, and partitioning prevent a direct numerical comparison. Nevertheless, the results support two conclusions: temporal sampling can be reduced without relevant loss in path-status performance, and evaluations based on random frame-level splitting produce considerably more optimistic estimates than protocols that preserve the temporal and spatial organization of the data. 


\section{Conclusion}\label{sec:conclusion}

This work presented \sensecv{}, an egocentric visual--inertial dataset for assistive navigation in indoor environments. The dataset contains 592 events collected across three locations, with videos, inertial signals, and metadata recorded using an open-source Android application. The protocol includes controlled events of free-path navigation, lateral deviation, and stopping, while accelerometer and gyroscope signals support temporal delimitation and label verification.

In the evaluation with Location C reserved exclusively for testing, MobileNetV4 achieved the best performance while also having the fewest parameters, reaching 95.58\% accuracy for path status and 89.97\% for deviation direction. Deviation classification exhibited greater variability across runs, indicating that this task is more difficult to generalize to environments not observed during training.

The experiments also showed that reducing the sampling rate from 30 to 10~FPS preserves path-status performance and may improve deviation classification when temporal separation is maintained. In contrast, random frame-level splitting produced artificially high results, confirming that correlation among samples can lead to overly optimistic performance estimates.

Future work will expand the diversity of locations, participants, and devices, evaluate temporal architectures, and explore the location and magnetometer signals currently preserved in the dataset.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{thebibliography}{00}

\bibitem{who2026} World Health Organization, ``Blindness and vision impairment,'' Feb. 10, 2026. [Online]. Available: \url{https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment}. [Accessed: Jul. 27, 2026].

\bibitem{casanova2025} E. Casanova, D. Guffanti, and L. Hidalgo, ``Technological advancements in human navigation for the visually impaired: A systematic review,'' \emph{Sensors}, vol. 25, no. 7, Art. no. 2213, 2025, doi: 10.3390/s25072213.

\bibitem{khan2022} D. Khan \emph{et al.}, ``Recent advances in vision-based indoor navigation: A systematic literature review,'' \emph{Computers \& Graphics}, vol. 104, pp. 24--45, 2022, doi: 10.1016/j.cag.2022.03.005.

\bibitem{gao2025} Y. Gao \emph{et al.}, ``A wearable obstacle avoidance device for visually impaired individuals with cross-modal learning,'' \emph{Nature Communications}, vol. 16, Art. no. 2857, 2025, doi: 10.1038/s41467-025-58085-x.

\bibitem{jadhav2025} A. Jadhav \emph{et al.}, ``AI Guide Dog: Egocentric path prediction on smartphone,'' \emph{Proceedings of the AAAI Symposium Series}, vol. 5, no. 1, pp. 220--227, 2025, doi: 10.1609/aaaiss.v5i1.35591.

\bibitem{medeiros2021} D. S. de Medeiros, T. H. Araújo, E. T. da Silva Júnior, and G. L. B. Ramalho, ``Using images to avoid collisions and bypass obstacles in indoor environments,'' in \emph{Anais Estendidos da XXXIV Conference on Graphics, Patterns and Images}, 2021, doi: 10.5753/sibgrapi.est.2021.20030.

\bibitem{figueiredo2024leakage} R. B. D. Figueiredo and H. A. Mendes, ``Analyzing information leakage on video object detection datasets by splitting images into clusters with high spatiotemporal correlation,'' \emph{IEEE Access}, vol. 12, pp. 47646--47655, 2024, doi: 10.1109/ACCESS.2024.3383047.

\bibitem{chitraranjan2025} C. Chitraranjan, V. Vipulananthan, and T. Sritharan, ``Vision-based collision warning systems with deep learning: A systematic review,'' \emph{Journal of Imaging}, vol. 11, no. 2, Art. no. 64, 2025, doi: 10.3390/jimaging11020064.

\bibitem{breve2020} F. A. Breve and C. N. Fischer, ``Visually impaired aid using convolutional neural networks, transfer learning, and particle competition and cooperation,'' in \emph{Proc. IJCNN}, 2020, pp. 1--8, doi: 10.1109/IJCNN48605.2020.9207606.

\bibitem{bastidas2025detectra} G. K. Bastidas-Guacho \emph{et al.}, ``Computer vision-based obstacle detection mobile system for visually impaired individuals,'' \emph{Multimodal Technologies and Interaction}, vol. 9, no. 5, Art. no. 48, 2025, doi: 10.3390/mti9050048.

\bibitem{plikynas2022} D. Plikynas, A. Indriulionis, A. Laukaitis, and L. Sakalauskas, ``Indoor-guided navigation for people who are blind: Crowdsourcing for route mapping and assistance,'' \emph{Applied Sciences}, vol. 12, no. 1, Art. no. 523, 2022, doi: 10.3390/app12010523.

\bibitem{loquercio2018} A. Loquercio, A. I. Maqueda, C. R. del-Blanco, and D. Scaramuzza, ``DroNet: Learning to fly by driving,'' \emph{IEEE Robotics and Automation Letters}, vol. 3, no. 2, pp. 1088--1095, 2018, doi: 10.1109/LRA.2018.2795643.

\bibitem{chang2022} Y. Chang \emph{et al.}, ``The HDIN dataset: A real-world indoor UAV dataset with multi-task labels for visual-based navigation,'' \emph{Drones}, vol. 6, no. 8, Art. no. 202, 2022, doi: 10.3390/drones6080202.

%\bibitem{samadzadeh2024} A. Samadzadeh \emph{et al.}, ``Amirkabir campus dataset: Real-world challenges and scenarios of visual inertial odometry for visually impaired people,'' \emph{arXiv:2401.03604}, 2024, doi: 10.48550/arXiv.2401.03604.

\bibitem{karnan2022scand} H. Karnan \emph{et al.}, ``Socially CompliAnt Navigation Dataset (SCAND): A large-scale dataset of demonstrations for social navigation,'' \emph{IEEE Robotics and Automation Letters}, vol. 7, no. 4, pp. 11807--11814, 2022, doi: 10.1109/LRA.2022.3184025.

\bibitem{liu2019travel} X. Liu, S. Zhang, J. Zeng, and F. Fan, ``Analysis and optimization strategy of travel system for urban visually impaired people,'' \emph{Sustainability}, vol. 11, no. 6, Art. no. 1735, 2019, doi: 10.3390/su11061735.

\bibitem{yan2020flobot} Z. Yan \emph{et al.}, ``Robot perception of static and dynamic objects with an autonomous floor scrubber,'' \emph{Intelligent Service Robotics}, vol. 13, no. 3, pp. 403--417, 2020, doi: 10.1007/s11370-020-00324-9.

\bibitem{manos2019gravity} A. Manos, I. Klein, and T. Hazan, ``Gravity-based methods for heading computation in pedestrian dead reckoning,'' \emph{Sensors}, vol. 19, no. 5, Art. no. 1170, 2019, doi: 10.3390/s19051170.

\bibitem{munozdiaz2017drift} E. Munoz Diaz, M. Caamano, and F. J. Fuentes-Sanchez, ``Landmark-based drift compensation algorithm for inertial pedestrian navigation,'' \emph{Sensors}, vol. 17, no. 7, Art. no. 1555, 2017, doi: 10.3390/s17071555.

\bibitem{park2010zerovelocity} S. K. Park and Y. S. Suh, ``A zero velocity detection algorithm using inertial sensors for pedestrian navigation systems,'' \emph{Sensors}, vol. 10, no. 10, pp. 9163--9178, 2010, doi: 10.3390/s101009163.

\bibitem{moenilssen2004gait} R. Moe-Nilssen and J. L. Helbostad, ``Estimation of gait cycle characteristics by trunk accelerometry,'' \emph{Journal of Biomechanics}, vol. 37, no. 1, pp. 121--126, 2004, doi: 10.1016/S0021-9290(03)00233-1.

\bibitem{rasouli2023gait} Z. Rasouli Kahaki, A. R. Safarpour, and H. Daneshmandi, ``The spatiotemporal gait parameters among people with visual impairment: A literature review study,'' \emph{Oman Journal of Ophthalmology}, vol. 16, no. 3, pp. 427--433, 2023, doi: 10.4103/ojo.ojo\_24\_23.

\bibitem{howard2017mobilenets} A. G. Howard \emph{et al.}, ``MobileNets: Efficient convolutional neural networks for mobile vision applications,'' \emph{arXiv:1704.04861}, 2017, doi: 10.48550/arXiv.1704.04861.

\bibitem{tan2019efficientnet} M. Tan and Q. V. Le, ``EfficientNet: Rethinking model scaling for convolutional neural networks,'' in \emph{Proc. ICML}, vol. 97, pp. 6105--6114, 2019.

\bibitem{he2016resnet} K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image recognition,'' in \emph{Proc. CVPR}, pp. 770--778, 2016, doi: 10.1109/CVPR.2016.90.

\bibitem{qin2024mobilenetv4} D. Qin \emph{et al.}, ``MobileNetV4: Universal models for the mobile ecosystem,'' in \emph{Computer Vision--ECCV 2024}, LNCS, vol. 15098, pp. 78--96, 2024, doi: 10.1007/978-3-031-73661-2\_5.

\end{thebibliography}
\vspace{12pt}

\end{document}
