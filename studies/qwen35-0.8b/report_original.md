# Qwen3.5-0.8B — 100 traced chats (original weights)

| Category | Correct | Confidence | Top guess from layer on | Reply tokens |
|---|---|---|---|---|
| math | 23/25 | 0.90 | 21.1 | 127.0 |
| facts | 18/20 | 0.75 | 21.9 | 5.8 |
| reasoning | 5/15 | 0.62 | 22.2 | 7.1 |
| language | 12/15 | 0.65 | 22.7 | 3.3 |
| code | 7/10 | 0.66 | 22.5 | 4.2 |
| commonsense | 13/15 | 0.65 | 22.2 | 5.9 |
| **all** | **78/100** | | | |

| | Answers | Confidence | Top guess from layer on | Reply tokens |
|---|---|---|---|---|
| right | 78 | 0.759 | 21.9 | 38.9 |
| wrong | 22 | 0.609 | 22.2 | 24.5 |

## Specialist neurons (top 5 per category)

| Category | Neuron | Effect size d | Active in category / elsewhere | Pushes words like |
|---|---|---|---|---|
| math | L23 #655 | -4.86 | 0.133 / 0.359 | '‑' '‐' '-' '-d' |
| math | L23 #223 | 4.63 | 0.628 / -0.158 | 'os' 'imini' '�' 'room' |
| math | L23 #1181 | -4.38 | -0.17 / -0.029 | ' –' '–' ' –,' '――' |
| math | L23 #141 | 4.36 | 0.418 / -0.13 | ' **„' ' entirely' ' "\\",' ' "/",' |
| math | L14 #699 | 4.21 | -0.054 / -0.15 | 'LEV' 'comb' 'levant' 'ifes' |
| facts | L12 #1829 | 2.76 | 0.078 / -0.0 | '或由' '复制或' '或被' ' cualquiera' |
| facts | L7 #3258 | 2.58 | 0.087 / 0.004 | '未知' 'unknown' '未知的' 'ULD' |
| facts | L11 #2265 | -2.55 | -0.164 / -0.017 | ' classics' ' cheered' ' Kors' 'adura' |
| facts | L5 #1373 | 2.53 | 0.08 / 0.014 | '.prop' 'ชัน' '四' 'ust' |
| facts | L9 #2209 | -2.53 | -0.071 / -0.006 | ' mill' ' contin' ' diab' ' es' |
| reasoning | L9 #1136 | -2.35 | -0.067 / -0.028 | 'iah' '被她' '汪洋' 'Find' |
| reasoning | L6 #122 | -2.2 | -0.122 / -0.023 | 'cores' '_given' '))*(' 'rej' |
| reasoning | L10 #501 | 2.11 | 0.085 / 0.031 | 'дини' '和特点' '老爸' ' aste' |
| reasoning | L16 #443 | -2.08 | -0.084 / -0.008 | 'ør' ':user' ':E' '»:' |
| reasoning | L10 #79 | -2.0 | -0.066 / 0.003 | '途' ' ines' 'utant' 'esa' |
| language | L8 #2757 | -2.92 | -0.073 / -0.029 | '不满' ' 아우' '不当' ' Chamber' |
| language | L17 #2921 | 2.9 | 0.089 / -0.05 | '代' ' Core' 'iard' '-era' |
| language | L13 #2395 | 2.88 | 0.071 / 0.013 | 'ivities' 'sat' '大放' ' bul' |
| language | L9 #101 | -2.86 | -0.073 / 0.005 | 'iki' '  ' 'atik' '落下' |
| language | L12 #2743 | 2.64 | 0.132 / 0.041 | '可能就' '可能需要' '可能对' '很可能' |
| code | L23 #165 | -5.62 | -1.344 / -0.202 | '際' 'omin' '-food' ' doğal' |
| code | L17 #603 | -5.12 | -0.207 / -0.041 | '随机' 'random' ' random' 'Random' |
| code | L15 #3279 | 4.97 | 0.215 / -0.0 | ' mil' 'ypo' '**-' 'ye' |
| code | L23 #939 | -4.52 | -1.548 / -0.335 | 'opping' 'ruptions' 'ído' 'agog' |
| code | L15 #1573 | 4.19 | 0.109 / -0.042 | 'Cer' ' cer' 'cer' '顷' |
| commonsense | L4 #2911 | -2.06 | -0.066 / -0.013 | 'xico' '一年以上' 'Ups' 'rite' |
| commonsense | L23 #1441 | 2.02 | -0.104 / -0.162 | '的结构' '的方向' '的条件' '的模式' |
| commonsense | L7 #2942 | -1.96 | -0.064 / -0.018 | 'istan' '工作目标' '危险因素' 'oes' |
| commonsense | L20 #115 | 1.93 | 0.113 / 0.029 | ' inner' ' ones' ' incoming' ' likelihood' |
| commonsense | L23 #2722 | -1.84 | -0.102 / 0.009 | ' token' ' tokens' ' Token' 'token' |

## Right vs wrong (78 right, 22 wrong in math, facts, reasoning, language, code, commonsense; chance level |d| ≈ 1.25)

| Neuron | d | Higher when | Pushes words like |
|---|---|---|---|
| L14 #3158 | -1.1 | wrong | 'lom' 'aat' '割' '尔顿' |
| L10 #1764 | 1.02 | right | 'succ' 'étel' ' succ' '详解' |
| L10 #75 | -1.0 | wrong | '滿足' ' definition' ' hiểu' ' račun' |
| L0 #990 | 0.98 | right | '但他的' '我的人' '通了' '这样的人' |
| L2 #3004 | -0.96 | wrong | 'imia' ' emer' '并发布' '并提出' |
| L13 #1567 | 0.91 | right | 'あげ' 'aid' ' rather' 'ues' |
| L11 #586 | 0.9 | right | '文山' 'neur' 'tika' '巡视员' |
| L10 #2664 | -0.9 | wrong | '但又' '但同时' '但仍然' '但仍' |
