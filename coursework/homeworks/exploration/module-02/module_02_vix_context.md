# Contexto — Module 02 Notes: VIX Calibration vs. Realized S&P 500 Volatility

Notebook exploratório (não é entrega de homework, é material de estudo próprio) investigando: **quão bem o VIX (índice de volatilidade implícita de 30 dias do S&P 500) prevê a volatilidade que de fato se realiza nos 21 pregões seguintes?**

Dados: `^GSPC` e `^VIX` via yfinance, 1990–presente (17/set/2026). Ferramentas: pandas, numpy, matplotlib, scikit-learn (Ridge regression).

---

## 1. Construção do alvo (realized volatility)

Duas definições de "volatilidade futura realizada" foram usadas ao longo do notebook:

- **`Future_std_vol`** (proxy inicial): desvio-padrão dos próximos 21 log-retornos diários, anualizado (`× √252 × 100`).
- **`Future_realized_vol`** (refinada, preferida): raiz da variância (sem subtrair a média) dos próximos 21 log-retornos ao quadrado — mais próxima do que o VIX teoricamente precifica.
- **`Trailing_realized_vol`**: a mesma fórmula da realized vol, mas usando os 21 dias **anteriores** (já conhecida no momento `t`) — serve de benchmark ingênuo ("naive").

Todo forecast é comparado estritamente fora da amostra: treino até 2022-11-30 (com purga dos últimos 21 dias pra evitar vazamento de dados pela sobreposição da janela futura), teste de 2023-01-03 em diante.

---

## 2. Gráfico 1 — VIX esperado vs. volatilidade realizada (2018–presente)

**O que mostra**: duas linhas (VIX em azul, volatilidade realizada nos 21 dias seguintes em laranja) de 2018 até set/2026, com a área entre elas pintada de verde (quando VIX superestimou) ou vermelho (quando subestimou).

**Leitura visual**: a área verde domina visualmente o gráfico inteiro — o VIX está **sistematicamente acima** da volatilidade realmente realizada na maior parte do tempo (viés positivo consistente). A única grande exceção visível é o pico de fev-mar/2020 (COVID): ali a volatilidade realizada disparou ACIMA do VIX por um curto período (faixa vermelha), quando o mercado realizou mais volatilidade do que o próprio VIX havia precificado antes do choque. Fora desse episódio, o padrão é: VIX cronicamente "caro" (superestima o risco futuro).

**Métricas (todo o período 2018–presente, sem calibração)**: Correlação 0,556 · MAE 6,653 · RMSE 9,593 · Bias +3,485 · R² 0,185 · n=2.168.

---

## 3. Calibração de viés (fixa e adaptativa)

Ideia: já que o VIX tem viés positivo sistemático, basta **subtrair esse viés** pra melhorar a previsão, sem precisar de nenhum modelo complexo.

- **Fixa**: viés médio estimado só com dados de treino (2018–2022), congelado, aplicado ao teste.
- **Adaptativa**: viés estimado com janela móvel de 252 dias, usando só erros já observáveis até aquela data (sem look-ahead).

**Resultados out-of-sample (2023-01-03 a 2026-08-18, n=909, alvo `Future_std_vol`)**:

| Modelo | Correlação | MAE | RMSE | Bias | R² |
|---|---:|---:|---:|---:|---:|
| VIX original | 0,431 | 5,559 | 7,044 | 3,987 | -0,286 |
| Calibração fixa | 0,431 | 3,663 | 5,874 | 0,879 | 0,106 |
| Calibração adaptativa | 0,404 | 3,893 | 6,095 | 0,580 | 0,037 |

Viés fixo estimado no treino: **3,107 pontos de volatilidade**. Melhoria de MAE: **34,1% (fixa)** e **30,0% (adaptativa)** vs. VIX cru. Note que o VIX original tem **R² negativo** (-0,286) — ou seja, prever "a média histórica" seria melhor do que usar o VIX cru diretamente; a calibração fixa já vira isso para R² positivo (0,106).

## 4. Gráfico 2 — Calibração fora da amostra (2023 em diante)

**O que mostra**: 4 linhas de 2023 a meados de 2026 — volatilidade futura realizada (preta, mais grossa), VIX original (azul claro), calibração fixa (roxo) e adaptativa (verde).

**Leitura visual**: o VIX azul-claro fica visivelmente **acima** da linha preta na maior parte do gráfico (confirma o viés visualmente). As linhas roxa e verde (calibradas) descem e ficam bem mais coladas à linha preta na maior parte do tempo — mas em dois picos de estresse (meados de 2024 e, principalmente, um pico grande no início de 2025, onde a realizada bate ~49% quase tão alto quanto o VIX original chega a ~52%) as três séries de expectativa (VIX, fixa, adaptativa) reagem em atraso/insuficientemente ao salto brusco da linha preta — a calibração ajuda no "nível médio", mas não resolve a defasagem de reação a choques rápidos.

## 5. Robustez anual e papel de 2020

Ano a ano (2023–2026), a calibração fixa melhora o MAE em **todos os 4 anos**, mas com magnitude bem desigual: 55,3% em 2023, 24,6% em 2024, 24,2% em 2025, 42,0% em 2026 — a melhoria é real e consistente (nunca negativa), mas não uniforme.

Excluindo 2020 do conjunto completo (não é o período de teste 2023+, é uma checagem separada em toda a amostra 2018+): a calibração adaptativa continua melhor que o VIX cru tanto incluindo 2020 (MAE 5,579 vs. 6,611) quanto excluindo (MAE 4,283 vs. 5,580) — ou seja, o ganho da calibração **não depende só do outlier de 2020**.

---

## 6. Refinamento metodológico + benchmark ingênuo

Agora usando o alvo `Future_realized_vol` (variância, mais correto teoricamente) e comparando também contra a **volatilidade histórica trailing** (o benchmark "ingênuo": só olhar pra trás).

**Resultados out-of-sample (2023+, n=909)**:

| Modelo | Correlação | MAE | RMSE | Bias | R² |
|---|---:|---:|---:|---:|---:|
| VIX original | 0,455 | 5,490 | 6,857 | 3,977 | -0,280 |
| Calibração fixa do VIX | 0,455 | 3,512 | 5,646 | 0,823 | 0,132 |
| Calibração adaptativa do VIX | 0,428 | 3,741 | 5,866 | 0,563 | 0,063 |
| Volatilidade trailing (naive) | 0,318 | 4,429 | 7,082 | 0,172 | -0,365 |

**Achado importante**: as duas versões calibradas do VIX (fixa e adaptativa) **batem o benchmark ingênuo** em MAE e R² — ou seja, o VIX calibrado carrega informação real além de "simplesmente olhar a volatilidade recente". Isso justifica usar opções/VIX como insumo, não só preço histórico.

## 7. Gráfico 3 — Benchmarks com o alvo refinado

**O que mostra**: realizada (preta), VIX original (azul claro), calibração fixa (roxo), volatilidade trailing/naive (laranja) — mesmo período 2023+.

**Leitura visual**: a linha laranja (naive/trailing) segue a preta com um **atraso temporal visível** — como é baseada em dados passados, ela reage DEPOIS que a volatilidade real já mudou, ficando sistematicamente "atrasada" nas subidas e descidas bruscas (visível claramente no pico de início de 2025, onde a laranja sobe DEPOIS da preta já ter caído). Já o roxo (VIX calibrado) reage praticamente ao mesmo tempo que a preta nesse mesmo pico, embora ainda subestime o tamanho exato do choque — isso ilustra bem por que o VIX (que é *forward-looking*, vem do mercado de opções) tem vantagem estrutural sobre um indicador puramente *backward-looking*.

---

## 8. Experimento supervisionado (Ridge Regression)

12 features conhecidas no momento da previsão (nível do VIX, variações de 1/5/21 dias, médias/desvio móveis do VIX, volatilidade trailing de 21/63 dias, retornos de 5/21 dias do S&P 500, drawdown de 252 dias). `TimeSeriesSplit` (5 splits, gap de 21 sessões) pra tuning; teste final 100% fora da amostra (2023+).

- Melhor `alpha` (regularização): **1000** (bem alto — indica forte regularização, sinal de que o modelo "quer" ficar simples).
- MAE de validação cruzada (treino): 6,834 — pior até que o MAE de teste do VIX cru (5,490)!
- **MAE de teste do Ridge: 4,136** — melhor que o VIX cru (5,490) e que o naive trailing (4,429), mas **pior que as duas calibrações simples de 1 parâmetro** (fixa: 3,512; adaptativa: 3,741).

**Achado central do notebook**: o modelo de 12 features com regularização **não supera a calibração de viés fixo, de um parâmetro só**. Mais complexidade não ajudou a generalizar melhor fora da amostra — um resultado clássico de "modelo simples vence" quando o sinal adicional das features é fraco/ruidoso relativo ao ganho de simplesmente corrigir o viés conhecido do VIX.

## 9. Gráfico 4 — Comparação final de modelos (2023+)

**O que mostra**: realizada (preta), VIX original (azul claro), calibração fixa (roxo), Ridge (verde) — mesmo eixo/período dos gráficos anteriores.

**Leitura visual**: a linha verde (Ridge) tende a ficar **entre** o VIX original e a calibração fixa a maior parte do tempo — mais alta que o roxo na maioria dos trechos calmos (sugerindo que o Ridge "aprendeu" um viés menor do que o ideal), mas reage de forma parecida ao roxo nos picos de estresse. Nenhuma das três séries de previsão acompanha totalmente a queda abrupta da linha preta logo após o pico de início de 2025 — todas ficam "presas" num patamar mais alto por vários dias antes de convergir de volta.

---

## 10. Interpretação dos coeficientes do Ridge (padronizados)

| Feature | Coeficiente |
|---|---:|
| VIX_std_21 | +1,149 |
| VIX_change_5 | +1,148 |
| VIX_expected (nível atual) | +1,128 |
| VIX_change_21 | +1,121 |
| SP500_drawdown_252 | **-1,012** |
| Trailing_vol_21 | +0,879 |
| VIX_mean_5 | +0,848 |
| SP500_return_5 | -0,535 |
| SP500_return_21 | -0,473 |
| Trailing_vol_63 | +0,401 |
| VIX_change_1 | +0,284 |
| VIX_mean_21 | +0,068 |

Leitura: o próprio nível/dispersão recente do VIX domina o modelo (esperado). O sinal mais interessante fora do VIX é o **drawdown de 252 dias** (negativo) — quanto mais o mercado está abaixo de sua máxima anual, maior a volatilidade futura prevista (faz sentido: quedas grandes tendem a coincidir com/preceder períodos mais voláteis). Retornos recentes negativos (5 e 21 dias) também empurram a previsão de volatilidade pra cima.

---

## Respostas às 4 perguntas de reflexão da conclusão (síntese própria, a partir dos números acima)

1. **A calibração fixa ainda reduz o MAE sob o alvo baseado em variância?** Sim — MAE cai de 5,490 (VIX cru) pra 3,512 (calibração fixa), uma redução de ~36%.
2. **Ela bate o forecast ingênuo de volatilidade trailing?** Sim — 3,512 (fixa) e 3,741 (adaptativa) contra 4,429 (naive); e o naive tem R² negativo (-0,365) enquanto as calibrações têm R² positivo.
3. **O Ridge melhora o MAE fora da amostra, ou a calibração de 1 parâmetro generaliza melhor?** A calibração simples generaliza melhor — Ridge fica em 4,136 (melhor que VIX cru e que o naive, mas **pior** que as duas calibrações de viés).
4. **As melhorias são consistentes entre anos e fora de períodos de crise?** Sim — a calibração fixa melhora o MAE em todos os 4 anos de teste (24%–55%), e o ganho da calibração adaptativa persiste mesmo excluindo 2020 do conjunto de treino/avaliação mais amplo.

## Limitações (do próprio notebook)

- Alvos de 21 sessões se sobrepõem, então os erros diários não são estatisticamente independentes.
- VIX é uma medida de volatilidade implícita neutra-a-risco, contém um prêmio de risco de variância — não foi desenhado pra ser um forecast pontual não-viesado.
- Correlação alta mede co-movimento, não calibração (nível certo).
- Resultados dependem da definição do alvo e do período de avaliação escolhidos.
- Ajustar repetidamente no conjunto de teste 2023+ invalidaria seu uso como teste "intocado".

## Arquivos de imagem extraídos (caso queira anexar às imagens na conversa com a outra IA)

1. `chart_1_cell8.png` — VIX vs. realizada, 2018–presente, área verde/vermelha.
2. `chart_2_cell15.png` — Calibração fixa/adaptativa fora da amostra, 2023+.
3. `chart_3_cell21.png` — Benchmarks (VIX calibrado vs. naive trailing), 2023+.
4. `chart_4_cell27.png` — Comparação final incluindo Ridge, 2023+.

Todos salvos em: `/tmp/claude-1000/-home-lucas-github-courses/334f4b85-af15-4d75-82d4-a756c5642f9f/scratchpad/vix_notebook/`
