# Cortes sugeridos por IMU: código, janelas e conhecimento do projeto

## Objetivo

Este documento reúne o caminho completo usado pelo SenseCV para:

1. sincronizar vídeo, acelerômetro e giroscópio;
2. detectar eventos físicos de movimento;
3. transformar esses eventos em janelas temporais de rotulagem;
4. resolver a classe e a direção com auxílio do manifesto;
5. sugerir os limites `start` e `end`;
6. efetuar o corte do vídeo;
7. selecionar imagens visualmente distintas dentro do corte;
8. expor o resultado pela interface, pelas APIs e pelos scripts de construção
   de datasets.

O código executável é a fonte autoritativa para o comportamento atual. A wiki
explica as decisões históricas e metodológicas, enquanto o documento de
orientação define o princípio experimental original [1-3].

## Fontes autoritativas do projeto

| Fonte | Papel |
|---|---|
| [`src/sensecv/app.py`](../src/sensecv/app.py) | Implementação principal da sincronização, detecção, sugestão, seleção de quadros, APIs e exportação |
| [`scripts/cut_manifest_deviation.py`](../scripts/cut_manifest_deviation.py) | Exportação em lote dos clipes presentes nos manifestos |
| [`scripts/make_datasetcortado.py`](../scripts/make_datasetcortado.py) | Dataset com o corte completo e todos os quadros e sensores |
| [`scripts/make_suggested_dinov2_dataset.py`](../scripts/make_suggested_dinov2_dataset.py) | Dataset de imagens escolhidas dentro dos cortes sugeridos |
| [`criterios_imu_video_orientando.pages.pdf`](../criterios_imu_video_orientando.pages.pdf) | Orientação metodológica original |
| [`crop-suggestion.md`](../docs/wiki/concepts/crop-suggestion.md) | Histórico e comportamento das modalidades de sugestão |
| [`imu-event-labeling.md`](../docs/wiki/concepts/imu-event-labeling.md) | Critérios de qualidade, eventos, janelas e confiança |
| [`api-routes.md`](../docs/wiki/entities/api-routes.md) | Contratos das APIs relacionadas |

## Princípio metodológico

A IMU e o vídeo exercem funções diferentes na rotulagem. Os sinais inerciais
localizam quando a resposta corporal começou e terminou. O vídeo estabelece o
conteúdo visual que motivou essa resposta. Portanto, o evento físico não deve
ser confundido com o alvo visual usado para treinamento.

Se \(T_1\) representa o início físico da resposta e \(T_2\) o seu término, a
imagem de interesse para uma política preditiva está, em geral, antes de
\(T_1\). Esse deslocamento temporal procura representar a informação que
estava disponível ao usuário antes da execução da ação [1-3].

De forma ampla, a transformação para eixos relacionados à gravidade, a análise
da periodicidade da marcha, a identificação de estados de movimento e o
controle de deriva são compatíveis com a literatura de processamento inercial
para navegação pedestre e análise da marcha no contexto da deficiência visual
[4-8]. As constantes presentes no código são parâmetros
do SenseCV e não valores universais fornecidos por essas referências.

## Visão geral do fluxo

```text
frames.json + accelerations.json + rotations.json
                         |
                         v
              _imu_event_series(idx)
        alinhamento temporal e sinais derivados
                         |
             +-----------+-----------+
             |                       |
             v                       v
    imu_quality_report(idx)   detect_imu_events(idx)
       qualidade global       desvio/redução/parada
                                     |
                                     v
                       suggest_deviation_cut(idx)
                    manifesto + eventos + fallbacks
                                     |
                         start, end, tipo, lado
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
           _ffmpeg_cut(...)                  ssim_frame_selection(...)
           grava o vídeo                       escolhe imagens
                  |                                     |
                  +------------------+------------------+
                                     v
                        exportação / API / dataset
```

É importante separar três operações:

- **detecção:** encontra eventos físicos e seus instantes;
- **sugestão:** escolhe qual evento e qual janela devem ser usados;
- **materialização:** corta o MP4 e, opcionalmente, seleciona quadros dentro
  desse intervalo.

## Instantes e janelas temporais

### Instantes

| Símbolo | Significado |
|---|---|
| \(T_0\) | instante em que a cena passa a exigir uma decisão; é um conceito visual e não é estimado diretamente pelo detector atual |
| \(T_1\) | início físico do evento identificado nos sinais inerciais |
| \(T_2\) | término do evento ou retorno a um estado de movimento estabilizado |
| \(\Delta\) | antecipação entre a janela visual e o início da resposta corporal |
| \(m\) | margem posterior usada na janela expandida |

### Janelas implementadas

| Nome | Intervalo | Finalidade |
|---|---|---|
| `acao` | \([T_1,T_2]\) | execução física da ação |
| `decisao` | \([T_1-\Delta,T_1]\) | evidência visual anterior à resposta; padrão do corte de desvio |
| `expandido` | \([T_1-\Delta,T_2+m]\) | contexto anterior e execução, com tolerância temporal |

Implementação em
[`_event_windows`, `app.py:2187`](../src/sensecv/app.py#L2187):

```python
def _event_windows(t1, t2, delta, duration):
    clamp = lambda v: round(min(max(0.0, v), duration), 3)
    return {
        'acao': [clamp(t1), clamp(t2)],
        'decisao': [clamp(t1 - delta), clamp(t1)],
        'expandido': [
            clamp(t1 - delta),
            clamp(t2 + IMU_EXPANDED_MARGIN_SEC),
        ],
    }
```

Todos os limites são restringidos à duração válida do clipe.

## Parâmetros que alteram as janelas e a detecção

As definições encontram-se em
[`app.py:1804`](../src/sensecv/app.py#L1804) e
[`app.py:2344`](../src/sensecv/app.py#L2344).

| Constante | Valor padrão | Efeito |
|---|---:|---|
| `IMU_EVENT_DELTA_SEC` | 1,0 s | antecipação da janela de decisão |
| `IMU_EVENT_PERSIST_SEC` | 0,4 s | persistência mínima de uma alteração lateral |
| `IMU_STOP_MIN_SEC` | 1,0 s | duração mínima do evento geral de parada |
| `IMU_STOP_ONSET_MIN_SEC` | 0,3 s | declarado, mas não referenciado pelo detector atual |
| `IMU_STOP_CONFIRM_SEC` | 0,8 s | declarado, mas não referenciado pelo detector atual |
| `IMU_STOP_TAIL_SEC` | 3,0 s | restringe a perda de cadência ao final do clipe |
| `IMU_DEVIATION_EDGE_MARGIN_SEC` | 1,0 s | rejeita desvios nas bordas do clipe |
| `IMU_STEP_WINDOW_SEC` | 2,0 s | janela da análise de periodicidade |
| `IMU_STEP_HOP_SEC` | 0,25 s | avanço entre janelas de periodicidade |
| `IMU_STEP_PERIODICITY` | 0,3 | evidência mínima de periodicidade |
| `IMU_EXPANDED_MARGIN_SEC` | 0,3 s | margem posterior da janela expandida |
| `IMU_DEVIATION_YAW_DEG_S` | 25,0 graus/s | referência para o sinal angular |
| `IMU_DEVIATION_LAT_VEL` | 0,035 | referência para o sinal lateral integrado |
| `IMU_DEVIATION_MERGE_GAP_SEC` | 1,5 s | intervalo máximo para unir fases opostas |
| `IMU_WALK_STD` | 0,5 m/s² | referência de energia para caminhada |
| `IMU_STOP_STD` | 0,25 m/s² | referência de energia para parada |
| `IMU_DECEL_RATIO` | 0,6 | razão relativa usada para desaceleração |
| `DEVIATION_CUT_WINDOW` | `decisao` | janela retornada por padrão |
| `TRIM_CUT_MARGIN_SEC` | 2,0 s | margem do modo `trim` |
| `TRIM_CUT_MIN_DURATION_SEC` | 8,0 s | duração mínima para aplicar a margem `trim` |

Esses valores podem ser substituídos por variáveis de ambiente com prefixo
`SENSECV_`, conforme as declarações do código.

## 1. Construção dos sinais alinhados

A função
[`_imu_event_series`, `app.py:1903`](../src/sensecv/app.py#L1903)
carrega as três séries, interpola os sensores nos timestamps dos quadros e
calcula os descritores usados pelas etapas seguintes.

Trecho central:

```python
secs = (ft - ft[0]) / 1e6
fps = float(1e6 / np.median(np.diff(ft)))

a_at_f = np.array(
    interp_at_times(at, av, ft.tolist()),
    dtype=np.float64,
)
amag = np.linalg.norm(a_at_f, axis=1)

W = min(len(ft), max(1, int(fps * 1.0)))
kernel = np.ones(W) / W
mean_mag = np.convolve(amag, kernel, mode='same')
var_mag = np.convolve((amag - mean_mag) ** 2, kernel, mode='same')
std_mag = np.sqrt(np.maximum(var_mag, 0.0))
```

### Base alinhada à gravidade

```python
g_smooth_w = max(1, int(round(fps * 0.7)))
g_dir = np.column_stack([
    _centered_rolling_mean(a_at_f[:, i], g_smooth_w)
    for i in range(3)
])
g_dir = g_dir / np.maximum(
    np.linalg.norm(g_dir, axis=1)[:, None],
    1e-6,
)

minus_z = np.zeros_like(g_dir)
minus_z[:, 2] = -1.0
fwd = np.cross(g_dir, np.cross(minus_z, g_dir))
fwd = fwd / np.maximum(
    np.linalg.norm(fwd, axis=1)[:, None],
    1e-6,
)
left = np.cross(g_dir, fwd)
```

O vetor `g_dir` define a vertical, `fwd` representa o eixo óptico projetado no
plano horizontal e `left` define a componente lateral.

### Descritor angular e descritor lateral

```python
yaw_deg_s = np.zeros(len(ft))
if len(rv):
    r_at_f = np.array(
        interp_at_times(rt, rv, ft.tolist()),
        dtype=np.float64,
    )
    yaw_rad_s = np.einsum('ij,ij->i', r_at_f, g_dir)
    smooth_w = max(1, int(round(fps * 0.2)))
    yaw_deg_s = np.degrees(
        np.array(_centered_rolling_mean(yaw_rad_s, smooth_w))
    )

lin = a_at_f - g_dir * np.linalg.norm(
    np.column_stack([
        _centered_rolling_mean(a_at_f[:, i], g_smooth_w)
        for i in range(3)
    ]),
    axis=1,
)[:, None]
lat_acc = np.einsum('ij,ij->i', lin, left)
lat_vel = np.cumsum(lat_acc) / fps
drift_w = max(1, int(round(fps * 2.5)))
lat_vel = lat_vel - np.array(
    _centered_rolling_mean(lat_vel, drift_w)
)
lat_vel = np.array(
    _centered_rolling_mean(
        lat_vel,
        max(1, int(round(fps * 0.6))),
    )
)
```

O descritor lateral é relativo. A remoção da média móvel reduz a componente de
baixa frequência acumulada pela integração, mas não transforma o resultado em
uma estimativa absoluta de velocidade.

## 2. Validação da qualidade inercial

[`imu_quality_report`, `app.py:2125`](../src/sensecv/app.py#L2125)
avalia:

- frequência mediana de amostragem;
- lacunas e irregularidade dos timestamps;
- saturação do acelerômetro e do giroscópio;
- fração do clipe compatível com caminhada;
- periodicidade e cadência;
- separabilidade entre estados de energia.

Código de decisão:

```python
checks = {
    'taxa_amostragem_100hz': acc_rate['median_hz'] >= 100.0,
    'sem_lacunas': acc_rate['gap_count'] == 0,
    'jitter_baixo': acc_rate['jitter_ratio'] < 0.5,
    'sem_saturacao': acc_sat < 0.01 and rot_sat < 0.01,
    'marcha_periodica': periodicity >= 0.3 and cadence_hz is not None,
    'tem_caminhada': walking_frac > 0.2,
    'estados_distinguiveis': (
        energy_range > 2.0
        or float(np.mean(stopped)) > 0.05
    ),
}

failed = [name for name, ok in checks.items() if not ok]
critical = {'sem_saturacao', 'tem_caminhada', 'marcha_periodica'}

if any(name in critical for name in failed):
    verdict = 'rejeitar'
elif failed:
    verdict = 'baixa_confianca'
else:
    verdict = 'aceitar'
```

O resultado de qualidade é usado na confiança dos eventos. Ele não substitui
a inspeção visual exigida pelo protocolo.

## 3. Detecção dos eventos físicos

[`detect_imu_events`, `app.py:2196`](../src/sensecv/app.py#L2196)
produz eventos `desvio`, `reducao` e `parada`, cada um com \(T_1\), \(T_2\),
janelas e confiança.

### Seleção automática do descritor lateral

```python
yaw_strength = (
    float(np.max(np.abs(yaw_deg_s))) / IMU_DEVIATION_YAW_DEG_S
    if len(yaw_deg_s)
    else 0.0
)

if yaw_strength >= 1.0:
    dev_signal = yaw_deg_s
    dev_threshold = IMU_DEVIATION_YAW_DEG_S
    dev_source = 'yaw'
else:
    dev_signal = series['lat_vel']
    dev_threshold = IMU_DEVIATION_LAT_VEL
    dev_source = 'lateral_velocity'
```

A escolha ocorre por clipe. A componente angular é utilizada quando contém
evidência suficiente; caso contrário, utiliza-se o descritor lateral derivado
do acelerômetro.

### Persistência e direção

```python
dev_mask = np.abs(dev_signal) > 0.6 * dev_threshold
for i0, i1 in _mask_runs(dev_mask.tolist()):
    if i1 - i0 + 1 < persist_n:
        continue

    segment = dev_signal[i0:i1 + 1]
    if float(np.max(np.abs(segment))) < dev_threshold:
        continue

    peak_local = int(np.argmax(np.abs(segment)))
    peak_i = i0 + peak_local
    sign = 1 if segment[peak_local] > 0 else -1
```

Os segmentos precisam ser temporalmente persistentes e atingir a magnitude de
referência. Fases próximas de sinais opostos podem ser unidas como partes de
uma única manobra de contorno. A polaridade da componente dominante define o
lado segundo a convenção da base `left = up × forward`.

### Estados de marcha

O mesmo detector usa `std_mag`, a variabilidade local da magnitude da
aceleração, para distinguir caminhada, redução de energia e parada:

```python
stop_mask = std_mag < IMU_STOP_STD
decel_mask = (
    (std_mag > IMU_STOP_STD)
    & (std_mag < IMU_DECEL_RATIO * baseline)
)
```

Cada evento recebe as três janelas produzidas por `_event_windows()`.

## 4. Qual evento se transforma em corte

[`suggest_deviation_cut`, `app.py:2644`](../src/sensecv/app.py#L2644)
é a função central. Ela não grava arquivos; retorna um dicionário contendo,
entre outros campos:

```json
{
  "found": true,
  "event_type": "desvio",
  "side": "LEFT",
  "window": "decisao",
  "start": 1.471,
  "end": 2.471,
  "t1": 2.471,
  "t2": 2.877,
  "confidence": "baixa",
  "label_source": "imu"
}
```

### Desvio

O detector:

1. reúne eventos `desvio`;
2. remove eventos incompatíveis com as margens ativas;
3. escolhe o evento de maior `strength`;
4. obtém `start` e `end` da janela configurada;
5. converte a direção em `LEFT` ou `RIGHT`.

Trecho decisivo:

```python
best = max(pool, key=lambda e: e.get('strength', 0.0))
windows = best.get('windows', {})
win = (
    windows.get(window)
    or windows.get('decisao')
    or [best['t1'], best['t2']]
)
start, end = float(win[0]), float(win[1])
side = side_override or {
    'esquerda': 'LEFT',
    'direita': 'RIGHT',
}.get(best.get('direction'), 'NONE')
```

### Recuperação orientada pelo manifesto

[`_manifest_kind_for_clip`, `app.py:2557`](../src/sensecv/app.py#L2557)
consulta o manifesto da coleta. Quando ele informa a classe, essa classe
define o ramo `desvio`, `parada` ou `livre`; para desvios, também define o lado.
Os sinais inerciais continuam responsáveis pelo posicionamento temporal.

Se o manifesto confirma um desvio, mas o detector principal não produz um
evento utilizável,
[`_desvio_window_from_signal`, `app.py:2577`](../src/sensecv/app.py#L2577)
procura a resposta lateral na direção conhecida. Se nem essa recuperação
encontrar um sinal adequado, o código cria uma janela aproximada em torno do
centro do clipe.

```python
if sig is not None:
    t1, t2, direction, strength = sig
    windows = _event_windows(t1, t2, delta_v, duration_clip)
    win = windows.get(window) or windows.get('decisao')
    start, end = float(win[0]), float(win[1])
else:
    mid = duration_clip / 2.0
    start = max(0.0, mid - delta_v / 2.0)
    end = min(duration_clip, start + delta_v)
```

### Fallback de parada

[`detect_stop_onset`, `app.py:2408`](../src/sensecv/app.py#L2408)
procura o instante em que a periodicidade das passadas desaparece:

- o clipe deve ser suficientemente longo;
- ao menos uma janela anterior deve apresentar periodicidade de passos;
- a última janela não pode continuar classificada como passos;
- a primeira janela após a última detecção de passos define \(T_1\);
- \(T_1\) deve estar na região final do clipe.

```python
centers, per = _step_rhythm(amag, fps)
stepping = per >= IMU_STEP_PERIODICITY

if not stepping.any():
    return None
if stepping[-1]:
    return None

stepping_idx = np.nonzero(stepping)[0]
onset = float(centers[int(stepping_idx[-1]) + 1])
if onset < duration - IMU_STOP_TAIL_SEC:
    return None
```

O corte usa a mesma janela de decisão anterior a \(T_1\). Quando o manifesto
confirma `parada`, mas o detector está incerto, o final do clipe é usado como
referência temporal. Embora `IMU_STOP_ONSET_MIN_SEC` e
`IMU_STOP_CONFIRM_SEC` continuem declarados entre as constantes, eles não são
consultados por `detect_stop_onset()` no estado atual do código.

### Fallback de caminhada livre

[`detect_free_walk_span`, `app.py:2439`](../src/sensecv/app.py#L2439)
verifica se existe ao menos uma janela com periodicidade de passos:

```python
_, per = _step_rhythm(amag, fps)
if len(per) == 0 or not (per >= IMU_STEP_PERIODICITY).any():
    return None
```

O ramo `_livre_cut()` retorna:

```python
dur = float(free['duration_s']) if free is not None else duration_clip
start, end = 0.0, dur
```

Portanto, o comportamento executável atual é **usar o clipe completo** para
`event_type='livre'`. Algumas passagens históricas da wiki descrevem “todo o
trecho útil de caminhada”; para reproduzir o dataset atual, deve prevalecer o
intervalo `[0, duration]` implementado no código.

### Ordem de decisão

Com classe conhecida pelo manifesto:

```python
if mk_type == 'livre':
    return (
        _livre_cut(force=True)
        or _parada_cut()
        or _desvio_cut(relax=True)
        or not_found
    )
if mk_type == 'parada':
    return (
        _parada_cut(force=True)
        or _livre_cut(force=True)
        or not_found
    )
if mk_type == 'desvio':
    return (
        _desvio_cut(side_override=mk_side, relax=True)
        or _parada_cut()
        or _livre_cut(force=True)
        or not_found
    )
```

Sem entrada no manifesto:

```python
return (
    _desvio_cut()
    or _parada_cut()
    or _livre_cut()
    or not_found
)
```

Assim, a ordem padrão é:

```text
desvio -> parada -> caminhada livre -> nenhum corte
```

## 5. Modo com margens

[`suggest_trim_cut`, `app.py:2355`](../src/sensecv/app.py#L2355)
é um invólucro de `suggest_deviation_cut`:

```python
def suggest_trim_cut(idx, margin=None):
    margin = TRIM_CUT_MARGIN_SEC if margin is None else float(margin)
    out = dict(suggest_deviation_cut(idx, cut_margin=margin))
    out.update({'mode': 'trim', 'margin': margin})
    if out.get('found'):
        out['trimmed'] = True
    return out
```

Nesse modo:

- a janela de desvio deve caber entre as margens do início e do final;
- um evento fora da região é descartado, e não deslocado;
- clipes curtos ignoram as margens;
- os ramos de parada e caminhada livre permanecem inalterados.

## 6. Diferença entre os modos chamados “lateral”

Há duas rotas históricas com nomes semelhantes:

| Superfície | Chamada atual |
|---|---|
| `GET /api/suggest/<idx>?mode=lateral` | `suggest_deviation_cut(idx)` |
| `export_set(..., mode='deviation')` | `suggest_deviation_cut(idx)` |
| `export_set(..., mode='lateral')` | `suggest_lateral_deviation(idx)` |

Assim, o botão “Desvio lateral” da interface usa o método baseado em eventos e
janelas \(T_1/T_2\), mas o modo literal `lateral` da função de exportação ainda
representa o detector histórico de inclinação. Para obter os cortes discutidos
neste documento em uma exportação em lote, deve-se usar
`mode='deviation'`.

## 7. Corte físico do vídeo

`suggest_deviation_cut()` apenas devolve os tempos. A gravação do novo MP4 é
feita por
[`_ffmpeg_cut`, `app.py:2872`](../src/sensecv/app.py#L2872):

```python
cmd = [
    ffmpeg,
    '-y',
    '-ss', f'{start:.3f}',
    '-noautorotate',
    '-i', src_video,
    '-t', f'{end-start:.3f}',
    '-map', '0:v:0',
    '-an',
    '-map_metadata', '0',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-crf', '23',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    out_video,
]
```

[`export_set`, `app.py:2969`](../src/sensecv/app.py#L2969)
coordena a exportação:

```python
if mode == 'deviation':
    sug = suggest_deviation_cut(idx)
elif mode == 'lateral':
    sug = suggest_lateral_deviation(idx)
else:
    sug = suggest_crop(idx, 'walking')

start, end = float(sug['start']), float(sug['end'])
selection = ssim_frame_selection(idx, start, end)

_ffmpeg_cut(
    clip_video_path(idx),
    start,
    end,
    out_video,
)
save_sensor_data(idx, start, end, out_folder)
save_ssim_selection(selection, out_folder)
```

O resultado inclui:

- MP4 recortado;
- sensores rebased para o início do corte;
- `ssim_selection.json`;
- vídeos de revisão;
- `sources.csv` com proveniência, limites e status.

## 8. Seleção de imagens dentro do corte

A janela temporal e a seleção de quadros são etapas independentes.
[`ssim_frame_selection`, `app.py:823`](../src/sensecv/app.py#L823)
percorre os quadros da janela sugerida e compara cada candidato com o último
quadro mantido.

```python
keep = (
    prev_rep is None
    or record_i == last_record_i
    or score is None
    or score < threshold
    or forced_gap
)

if keep:
    selected.append({
        'source_index': int(source_index),
        'frame_id': int(frame.get('frame_id', source_index)),
        'time_s': round(float(time_s), 3),
        'ssim_prev': (
            None if score is None else round(float(score), 5)
        ),
    })
```

Propriedades:

- o primeiro e o último quadro são preservados;
- um quadro suficientemente diferente é preservado;
- um intervalo máximo força cobertura temporal mesmo em trechos estáticos;
- a seleção pode usar `ssim`, `dinov2`, `lpips` ou `vif`;
- a API informa `frames_before` e `frames_after`.

Essa seleção **não altera** `start` e `end`. Ela apenas decide quais imagens
dentro do intervalo serão usadas por um dataset ou exibidas na revisão.

## 9. APIs

### Sugestão

[`GET /api/suggest/<idx>`, `app.py:4727`](../src/sensecv/app.py#L4727):

```python
if mode == 'lateral':
    return jsonify(_suggestion_with_ssim(
        idx,
        suggest_deviation_cut(idx),
        threshold=threshold,
        metric=metric,
    ))

if mode == 'trim':
    return jsonify(_suggestion_with_ssim(
        idx,
        suggest_trim_cut(idx),
        threshold=threshold,
        metric=metric,
    ))
```

`_suggestion_with_ssim()` acrescenta as contagens da seleção visual ao
dicionário do corte, mas não grava o vídeo.

### Eventos

[`GET /api/imu-events/<idx>`, `app.py:4774`](../src/sensecv/app.py#L4774)
retorna todos os eventos, suas janelas e a avaliação de confiança:

```python
delta = float(
    request.args.get('delta')
    or IMU_EVENT_DELTA_SEC
)
return jsonify(detect_imu_events(idx, delta=delta))
```

### Inspeção de desvios

[`POST /api/inspect-deviation`, `app.py:4617`](../src/sensecv/app.py#L4617)
chama
[`export_deviation_set`, `app.py:4350`](../src/sensecv/app.py#L4350).
Essa função:

1. calcula o corte com `suggest_deviation_cut`;
2. mantém somente `side in {'LEFT', 'RIGHT'}`;
3. executa `export_set(..., mode='deviation')`;
4. produz um vídeo concatenado de validação.

Logo, os fallbacks `parada` e `livre` podem ser exportados por `export_set`,
mas são excluídos especificamente do vídeo de inspeção de desvios.

## 10. Comportamento da interface

O carregamento de um clipe executa
[`runAutoSuggest`, `index.html:1967`](../src/sensecv/templates/index.html#L1967):

```javascript
async function runAutoSuggest(){
  const reqIdx = currentIdx;
  for (const mode of ['trim', 'walking']) {
    const found = await runSuggest(mode, {auto:true});
    if (reqIdx !== currentIdx || found) return;
  }
}
```

Assim, a interface tenta primeiro o corte de eventos com margens. Somente
quando esse caminho retorna `found:false` é que a sugestão geral de caminhada
é solicitada. Os três botões permitem executar manualmente:

- `walking`: sugestão de caminhada;
- `lateral`: `suggest_deviation_cut()` sem as margens do modo `trim`;
- `trim`: `suggest_deviation_cut()` com as margens.

[`cutTypeInfo`, `index.html:1670`](../src/sensecv/templates/index.html#L1670)
exibe o tipo resolvido como desvio, parada, caminhada livre ou caminhada. A
interface também oferece as três janelas de cada evento no painel “Eventos
IMU”, permitindo que o operador substitua o intervalo sugerido.

## 11. Scripts que consomem os cortes

### `cut_manifest_deviation.py`

[`scripts/cut_manifest_deviation.py`](../scripts/cut_manifest_deviation.py)
seleciona somente clipes com linha de manifesto, agrupa-os por coleta e chama:

```python
summary = app.export_set(
    indices,
    out_dir,
    mode='deviation',
    verbose=True,
)
```

Ele grava cortes, sensores, seleção visual, revisão e `sources.csv`.

### `make_datasetcortado.py`

[`scripts/make_datasetcortado.py:77`](../scripts/make_datasetcortado.py#L77)
usa o mesmo `suggest_deviation_cut`, mas não aplica seleção de similaridade:

```python
sug = app.suggest_deviation_cut(idx)
start, end = float(sug['start']), float(sug['end'])

app._ffmpeg_cut(
    app.clip_video_path(idx),
    start,
    end,
    out_video,
)
app.save_sensor_data(idx, start, end, out_folder)
```

Consequência: todos os quadros do corte aparecem uma vez no clipe resultante.

### `make_suggested_dinov2_dataset.py`

[`scripts/make_suggested_dinov2_dataset.py:199`](../scripts/make_suggested_dinov2_dataset.py#L199)
também começa pelo mesmo corte:

```python
sug = app.suggest_deviation_cut(idx)
rows, summary = sample_clip(
    metric,
    idx,
    display,
    app.clip_video_path(idx),
    sug,
    images_dir,
    threshold,
    args.jpeg_quality,
)
```

Nesse caso, `sample_clip()` aplica a métrica DINOv2 para escolher imagens
visualmente distintas dentro de `[start,end]`.

### `imu_event_report.py`

[`scripts/imu_event_report.py`](../scripts/imu_event_report.py)
executa `detect_imu_events()` em lote e grava relatórios de qualidade e eventos.
Ele é útil para auditoria do detector, mas não materializa cortes.

## 12. Observações registradas na wiki

Os valores abaixo são auditorias históricas do estado dos dados, e não regras
do algoritmo:

- nos 32 clipes do manifesto IFCE usados na verificação de direção, foram
  registrados 26 acertos, três eventos ausentes e três direções incorretas;
  duas das três direções incorretas receberam baixa confiança;
- nesses clipes, a estabilização da câmera reduziu a informação angular e o
  descritor lateral foi selecionado automaticamente;
- uma varredura histórica da wiki registra zero cortes de parada em 389
  clipes; esse número deve ser reavaliado após mudanças no detector de perda de
  cadência;
- uma varredura histórica de 389 clipes registrou 162 desvios, zero paradas,
  97 caminhadas livres e 130 casos sem evento utilizável;
- na verificação inicial do modo `trim`, clipes curtos produziram o mesmo
  intervalo que o corte sem margens, enquanto os desvios de clipes longos
  permaneceram dentro da região permitida;
- a seleção visual preserva o primeiro e o último quadro e força cobertura
  periódica mesmo quando imagens consecutivas são muito semelhantes.

Esses números constam em `imu-event-labeling.md` e `crop-suggestion.md` [2,3].
Eles devem ser recalculados se as fontes, os parâmetros ou o conjunto de clipes
forem alterados.

## 13. Proveniência e rótulos

O corte sugerido preserva:

- `source_display` ou `source_clip`;
- início e fim no vídeo original;
- \(T_1\) e \(T_2\), quando disponíveis;
- `event_type`: `desvio`, `parada` ou `livre`;
- `side`: `LEFT`, `RIGHT` ou `NONE`;
- `label_source`: `manifesto` ou `imu`;
- confiança e intensidade, quando provenientes de um evento;
- número de quadros antes e depois da seleção visual;
- parâmetros da métrica usada.

O manifesto tem precedência sobre a IMU para a classe semântica e para o lado
do desvio. Isso evita que uma estimativa inercial substitua uma anotação
revisada. A IMU continua sendo usada para posicionar temporalmente o corte.

## 14. Regras que devem permanecer claras

1. A IMU localiza a resposta corporal; o vídeo determina a causa visual.
2. A janela `decisao` antecede \(T_1\) e é o padrão para treinamento preditivo.
3. A janela `acao` contém a execução e não é equivalente à janela de decisão.
4. O manifesto prevalece na classe e no lado quando existe.
5. Sem desvio, a cadeia tenta parada e depois caminhada livre.
6. O ramo atual de caminhada livre retorna o clipe completo.
7. `suggest_deviation_cut()` sugere tempos; `_ffmpeg_cut()` grava o MP4.
8. A seleção SSIM/DINOv2 acontece depois da escolha temporal.
9. `export_set(mode='deviation')` usa o método atual de eventos.
10. `export_set(mode='lateral')` ainda usa o detector lateral histórico.
11. O vídeo de inspeção mantém somente desvios `LEFT` e `RIGHT`.
12. `datasetcortado` preserva todos os quadros; o dataset DINOv2 aplica seleção.

## Referências

### Documentação interna

1. SenseCV. *Critérios para validação da IMU e seleção de trechos de vídeo:
   orientações para rotulagem de caminhada em primeira pessoa com dados
   inerciais*. Documento interno,
   [`criterios_imu_video_orientando.pages.pdf`](../criterios_imu_video_orientando.pages.pdf).
2. SenseCV Wiki. *IMU event labeling*,
   [`docs/wiki/concepts/imu-event-labeling.md`](../docs/wiki/concepts/imu-event-labeling.md).
3. SenseCV Wiki. *Crop suggestion*,
   [`docs/wiki/concepts/crop-suggestion.md`](../docs/wiki/concepts/crop-suggestion.md).

### Fundamentação acadêmica

4. A. Manos, I. Klein e T. Hazan. “Gravity-Based Methods for Heading
   Computation in Pedestrian Dead Reckoning.” *Sensors*, 19(5), 1170, 2019.
   [https://doi.org/10.3390/s19051170](https://doi.org/10.3390/s19051170).
5. R. Moe-Nilssen e J. L. Helbostad. “Estimation of Gait Cycle
   Characteristics by Trunk Accelerometry.” *Journal of Biomechanics*, 37(1),
   121-126, 2004.
   [https://doi.org/10.1016/S0021-9290(03)00233-1](https://doi.org/10.1016/S0021-9290(03)00233-1).
6. S. K. Park e Y. S. Suh. “A Zero Velocity Detection Algorithm Using
   Inertial Sensors for Pedestrian Navigation Systems.” *Sensors*, 10(10),
   9163-9178, 2010.
   [https://doi.org/10.3390/s101009163](https://doi.org/10.3390/s101009163).
7. E. Munoz Diaz, M. Caamano e F. J. Fuentes-Sanchez. “Landmark-Based Drift
   Compensation Algorithm for Inertial Pedestrian Navigation.” *Sensors*,
   17(7), 1555, 2017.
   [https://doi.org/10.3390/s17071555](https://doi.org/10.3390/s17071555).
8. Z. Rasouli Kahaki, A. R. Safarpour e H. Daneshmandi. “The Spatiotemporal
   Gait Parameters Among People with Visual Impairment: A Literature Review
   Study.” *Oman Journal of Ophthalmology*, 16(3), 427-433, 2023.
   [https://doi.org/10.4103/ojo.ojo_24_23](https://doi.org/10.4103/ojo.ojo_24_23).

## Nota de manutenção

As linhas citadas correspondem ao estado do repositório em 28 de julho de
2026. Caso `src/sensecv/app.py` seja alterado, devem ser revisados primeiro:

- parâmetros e nomes dos modos;
- ordem dos fallbacks;
- precedência do manifesto;
- fórmulas de `_event_windows`;
- comportamento de `_livre_cut`;
- seleção de quadros e formatos dos payloads;
- consumidores em `scripts/`.
