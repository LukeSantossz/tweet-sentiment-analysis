# TASKS.md — Registro de Tasks para Implementacao

> **Este arquivo e o ponto de entrada obrigatorio para qualquer implementacao.**
> Nenhum agente de IA pode modificar a codebase sem uma task formalmente registrada aqui.
> Consulte `.claude/rules/00-trava-seguranca.md` para as regras completas.

---

## Como Usar

1. Copie o template da Secao "Template de Task" abaixo.
2. Preencha todos os campos obrigatorios (marcados com `!`).
3. Adicione a task preenchida na Secao "Tasks Ativas".
4. Inicie a sessao com o agente informando o modo de operacao desejado (Desenvolvimento, Review ou Tutor).
5. Ao concluir, mova a task para "Tasks Concluidas" com o resultado preenchido.

---

## Template de Task

```markdown
### TASK-[NNN]
- **Status:** pendente | em andamento | concluida | descartada | revertida
- **Modo:** desenvolvimento | review | tutor
- **Complexidade:** patch | minor | major
- **Data de criacao:** [YYYY-MM-DD]

#### Objetivo (!obrigatorio)
[Descreva de forma direta o que precisa ser feito. Uma frase clara.
Teste: se alguem ler apenas esta linha, entende o que sera entregue?]

#### Contexto (!obrigatorio)
[Por que essa mudanca e necessaria? Qual problema resolve?
Se houver link de issue, PR, ou card de projeto, inclua aqui.]

#### Escopo Tecnico (!obrigatorio)
- **Arquivos/modulos envolvidos:** [listar os arquivos ou areas que serao tocados]
- **Dependencias necessarias:** [novas dependencias ou "nenhuma"]
- **Impacto em funcionalidades existentes:** [descrever ou "nenhum"]

#### Criterios de Aceite (!obrigatorio)
[Liste as entregas concretas que definem a task como concluida.
Cada criterio deve ser verificavel — sim ou nao, passou ou nao passou.]
- [ ] [Criterio 1]
- [ ] [Criterio 2]
- [ ] [Criterio 3]

#### Restricoes (opcional)
[Limitacoes tecnicas, de tempo, de escopo, ou decisoes ja tomadas que o agente deve respeitar.]

#### Referencias (opcional)
[Links de documentacao, PRs anteriores, issues relacionadas, artigos tecnicos relevantes.]

#### Log de Andamento (atualizado pelo agente)
> Registro cronologico do progresso da task. O agente adiciona uma entrada a cada sessao em que a task for trabalhada, incluindo sessoes onde houve travamento ou interrupcao. Nunca remova entradas anteriores.

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| —    | —      | —              | —               |

#### Resultado (preenchido ao concluir)
- **Data de conclusao:** [YYYY-MM-DD]
- **Branch:** [nome da branch utilizada]
- **Commit(s):** [hash ou mensagem]
- **Avaliacao pos-implementacao:** [aprovado / aprovado com ressalvas / reprovado]
- **Observacoes:** [notas relevantes para futuras tasks]
```

### Classificacao de Complexidade

A complexidade determina o nivel de cerimonia na avaliacao pos-implementacao (ver `.claude/rules/04-avaliacao-pos.md`):

| Nivel | Quando usar | Exemplos |
|-------|-------------|----------|
| **patch** | Mudanca trivial, sem risco de efeito colateral | Renomear variavel, corrigir typo, ajustar espacamento, remover import nao utilizado |
| **minor** | Mudanca localizada em um modulo, risco baixo | Implementar funcao isolada, corrigir bug em um arquivo, adicionar teste |
| **major** | Mudanca estrutural, multiplos arquivos, risco de impacto em cascata | Nova feature com multiplos modulos, refatoracao arquitetural, migracao de dependencia |

---

## Tasks Ativas

> Tasks em andamento ou pendentes de implementacao. O agente so pode trabalhar em tasks listadas aqui.
> **Regra de ordenacao:** A primeira task listada e a task ativa. O agente trabalha nela ate conclusao, descarte ou bloqueio explicito pelo usuario. Para mudar a prioridade, o usuario reordena as tasks nesta secao.

### TASK-025
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-09

#### Objetivo
Corrigir organizacao de tasks, dependencias faltantes, emojis multi-codepoint no Rust e arquivo nul acidental.

#### Contexto
Auditoria identificou: (1) 5 tasks concluidas na secao errada; (2) polars faltando em requirements.txt; (3) divergencia de emojis multi-codepoint entre Python e Rust; (4) arquivo `nul` criado acidentalmente na raiz.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.claude/tasks.md`, `requirements.txt`, `rust/tweet-preprocessor/src/main.rs`, `rust/tweet-preprocessor/Cargo.toml`, `nul` (remocao)
- **Dependencias necessarias:** unicode-segmentation (Rust crate)
- **Impacto em funcionalidades existentes:** Rust CLI passa a processar emojis multi-codepoint corretamente

#### Criterios de Aceite
- [x] Tasks concluidas (TASK-017, 018, 019, 020, 024) movidas para secao Tasks Concluidas
- [x] polars e numpy adicionados ao requirements.txt
- [x] Funcao handle_emojis do Rust corrigida para multi-codepoint (unicode-segmentation)
- [x] Arquivo nul removido do diretorio raiz
- [x] Testes Rust passando com emojis multi-codepoint (7 testes ok)

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-09 | 1 | requirements.txt, main.rs, Cargo.toml, nul removido, 7 testes Rust, 20 testes Python, benchmark parity OK, tasks reorganizadas | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-09
- **Branch:** feat/TASK-020-024-rust-preprocessing
- **Commit(s):** pendente
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Paridade Python/Rust validada via benchmark (7.9x speedup em 10k tweets). Emojis multi-codepoint (flags, skin tones, ZWJ) agora processados corretamente via grapheme clusters.

---

### TASK-021
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-05-07

#### Objetivo
Implementar script de batch inference otimizado para processar 1.6M tweets preprocessados com GPU.

#### Contexto
Apos preprocessing em Rust (TASK-020), o pipeline Python recebe Parquet limpo e executa inferencia em batch. O script deve maximizar throughput via DataLoader otimizado, batching adequado e gerenciamento de memoria GPU. Depende de TASK-007 (modelo fine-tuned) e TASK-020 (preprocessing Rust).

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `src/batch_inference.py` (novo)
- **Dependencias necessarias:** `polars` (leitura Parquet), `torch`, `transformers`
- **Impacto em funcionalidades existentes:** nenhum (componente novo)

#### Criterios de Aceite
- [ ] Script aceita `--input` (Parquet preprocessado) e `--output` (Parquet com predicoes)
- [ ] DataLoader com batching otimizado (batch_size configuravel, default 64)
- [ ] Inferencia em GPU com `torch.no_grad()` e `model.eval()`
- [ ] Barra de progresso com estimativa de tempo (tqdm)
- [ ] Output contem colunas: text_original, label, score, processing_time_ms
- [ ] Metricas de throughput ao final: tweets/segundo, tempo total

#### Restricoes
- Carregar modelo uma unica vez no inicio
- Nao carregar dataset inteiro em memoria — usar chunks/streaming se necessario
- Liberar memoria GPU entre batches se necessario (torch.cuda.empty_cache)

#### Referencias
- https://huggingface.co/docs/transformers/main_classes/pipelines
- https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-022
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-07

#### Objetivo
Executar benchmark comparativo Python vs Rust preprocessing e documentar resultados no README.

#### Contexto
Validar o ganho de performance do preprocessing em Rust (TASK-020) vs Python puro. O benchmark deve ser reproduzivel e os resultados documentados no README como evidencia de decisao tecnica. Depende de TASK-020 concluida.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `benchmarks/preprocessing_benchmark.py` (novo), `README.md`
- **Dependencias necessarias:** `hyperfine` (CLI benchmark) ou `time` manual
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] Script de benchmark com dataset de teste (10k, 100k, 1M tweets)
- [ ] Medicao de tempo para Python (`src/preprocessing.py`) e Rust (`rust/tweet-preprocessor`)
- [ ] Tabela comparativa no README: tweets/segundo, speedup factor
- [ ] Validacao de paridade funcional: output identico para ambos

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-023
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-07

#### Objetivo
Atualizar arquitetura do projeto e README para refletir pipeline de escala com Rust + Python.

#### Contexto
Apos implementacao do preprocessing Rust e batch inference, o README e diagrama de arquitetura devem ser atualizados para refletir a nova arquitetura. Depende de TASK-020, TASK-021 e TASK-022 concluidas.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `README.md`, `.claude/registry.md`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] Diagrama Mermaid atualizado com fluxo Rust -> Python
- [ ] Secao de arquitetura explicando a decisao Rust para preprocessing
- [ ] Resultados de benchmark incluidos na secao de decisoes tecnicas
- [ ] Instrucoes de execucao do pipeline completo (Rust + Python)

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-007
- **Status:** em andamento
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-03-29

#### Objetivo
Implementar e executar o fine-tuning do modelo twitter-roberta-base-sentiment usando a Trainer API do HuggingFace.

#### Contexto
O baseline zero-shot (TASK-006) atingiu 70% accuracy e 0.71 F1 macro. O fine-tuning visa superar esses numeros treinando o modelo no dataset TweetEval sentiment. O script ja foi implementado e commitado, mas o treinamento nunca foi executado — nenhum checkpoint existe em outputs/.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `src/training.py`, `tests/test_training.py`, `outputs/`
- **Dependencias necessarias:** transformers, datasets, scikit-learn, accelerate (ja instaladas)
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Script de treinamento implementado com Trainer API
- [x] Hiperparametros documentados no script
- [x] Testes unitarios do modulo de training implementados
- [ ] Script de treinamento executado sem erros ate o fim
- [ ] Melhor checkpoint salvo em disco
- [ ] Loss e metricas de validacao monitorados por epoca

#### Referencias
- https://huggingface.co/docs/transformers/training

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-03-29 | 1 | Script src/training.py implementado com Trainer API | em andamento |
| 2026-03-30 | 2 | Testes unitarios adicionados em tests/test_training.py | em andamento |
| 2026-05-07 | 3 | Dependencias instaladas, treinamento cancelado (CPU only ~25h) — aguardando ambiente com GPU | em andamento |

#### Resultado (preenchido ao concluir)
- **Data de conclusao:** —
- **Branch:** dev
- **Commit(s):** 8344a3c feat(training): add fine-tuning script with Trainer API, 0dac9aa test(training): add unit tests for training module
- **Avaliacao pos-implementacao:** —
- **Observacoes:** —

---

### TASK-008
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-03-29

#### Objetivo
Executar avaliacao comparativa entre o modelo fine-tuned e o baseline zero-shot, com metricas e analise de erros documentadas.

#### Contexto
Apos o fine-tuning (TASK-007), e necessario quantificar o ganho real sobre o baseline (70% acc, 0.71 F1). A comparacao inclui metricas por classe, matrizes de confusao e analise de erros. Depende de TASK-007 concluida.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `notebooks/05_evaluation.ipynb`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] Tabela comparativa baseline vs fine-tuned com accuracy e F1 macro
- [ ] Ganho percentual em F1 macro calculado e documentado
- [ ] Matrizes de confusao de ambos os modelos plotadas lado a lado
- [ ] Analise de erros com ao menos uma hipotese por classe divergente

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-010
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-05-01

#### Objetivo
Implementar endpoint de predicao REST com FastAPI usando o modelo fine-tuned.

#### Contexto
Camada de servico do pipeline. O endpoint POST /predict recebe texto de tweet e retorna label de sentimento e score de confianca. Modelo carregado uma unica vez na inicializacao via lifespan context. Depende de TASK-008 concluida (checkpoint e metricas disponiveis).

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `src/app/main.py`, `src/app/schemas.py`
- **Dependencias necessarias:** fastapi, uvicorn[standard], pydantic
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] Endpoint POST /predict retorna {"label": ..., "score": ...} com HTTP 200
- [ ] Endpoint GET /health retorna {"status": "ok"} com HTTP 200
- [ ] Modelo e tokenizer carregados uma unica vez na inicializacao
- [ ] Pipeline de pre-processamento aplicado ao input antes da inferencia
- [ ] Validacao do payload com Pydantic: campo text, string nao vazia

#### Referencias
- https://fastapi.tiangolo.com/advanced/events/
- https://huggingface.co/docs/transformers/main_classes/pipelines

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-011
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-01

#### Objetivo
Implementar interface de demonstracao interativa com Gradio conectada ao endpoint FastAPI.

#### Contexto
Camada de demonstracao publica do projeto para portfolio. O usuario insere texto de tweet e recebe label de sentimento com score via componente gr.Label. Depende de TASK-010 concluida (endpoint disponivel).

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `frontend/app.py`
- **Dependencias necessarias:** gradio, requests
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] Interface acessivel em localhost:7860 com python frontend/app.py
- [ ] Campo de input de texto e botao de submit funcionando
- [ ] Resposta exibindo label e score retornados pelo endpoint FastAPI
- [ ] Tres exemplos pre-carregados (um por label: negative, neutral, positive)
- [ ] URL da API configuravel via variavel de ambiente API_URL

#### Referencias
- https://www.gradio.app/docs/gradio/blocks
- https://www.gradio.app/docs/gradio/label

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-012
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-05-01

#### Objetivo
Containerizar aplicacao com Dockerfile e docker-compose para orquestrar servicos api e frontend.

#### Contexto
Compose deve expor portas, injetar variaveis de ambiente e garantir dependencia de inicializacao entre servicos. Checkpoint do modelo montado como volume. Depende de TASK-010 e TASK-011 concluidas.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- **Dependencias necessarias:** Docker Engine >= 24, docker-compose v2
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] docker-compose up sobe API e frontend sem erros em ambiente limpo
- [ ] GET /health responde com HTTP 200 apos docker-compose up
- [ ] Frontend Gradio acessivel em localhost:7860 via compose
- [ ] Checkpoint do modelo acessivel dentro do container via volume
- [ ] .dockerignore configurado para excluir venv/, __pycache__/, notebooks/

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-009
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-04-07

#### Objetivo
Configurar pipeline de CI com GitHub Actions para lint, testes automatizados e verificacao de build Docker.

#### Contexto
Garantir que todo PR e push para main passe por lint (Ruff), testes (pytest) e build do container antes do merge. Depende de TASK-012 concluida (Dockerfile existente para o job docker-build).

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.github/workflows/ci.yml`, `pyproject.toml`
- **Dependencias necessarias:** ruff, pytest-cov
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] Pipeline verde em push para main e em PRs abertos
- [ ] Job de lint via Ruff sem erros
- [ ] Job de pytest com relatorio de cobertura (minimo 70%, nao bloqueante)
- [ ] Job de docker build sem erros (sem push)
- [ ] Badge de CI visivel no README

#### Referencias
- https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python
- https://docs.astral.sh/ruff/configuration/

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

### TASK-013
- **Status:** pendente
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-01

#### Objetivo
Finalizar README.md profissional para portfolio com arquitetura, resultados, instrucoes de uso e badge de CI.

#### Contexto
Task final do projeto. O README deve ser suficiente para que um recrutador ou engenheiro externo compreenda o projeto, rode localmente e avalie as decisoes tecnicas. Depende de TASK-008 (metricas), TASK-009 (badge CI), TASK-010/011/012 (instrucoes Docker e API).

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `README.md`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [ ] README com secoes: Overview, Architecture, Results, Getting Started, API Reference, Docker, Project Structure
- [ ] Tabela de resultados com accuracy e F1 macro (baseline vs fine-tuned)
- [ ] Badge de CI do GitHub Actions presente e verde
- [ ] Instrucoes de instalacao testadas em ambiente limpo
- [ ] Instrucoes de execucao via docker-compose up testadas
- [ ] Exemplo de chamada a API via curl e via Python requests

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| — | — | — | — |

---

## Tasks Concluidas

> Tasks finalizadas. Movidas para ca apos conclusao e atualizacao do Registro de Projeto (`registry.md`). Nunca remova entradas — o historico e cumulativo.

### TASK-024
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-07

#### Objetivo
Corrigir findings do review adversarial da TASK-020 e completar fluxo CRURA pendente.

#### Contexto
Review adversarial (Codex) identificou 7 findings na TASK-020, incluindo 2 de alta severidade. Adicionalmente, o .gitignore nao inclui Rust build artifacts e os commits da TASK-020 estao pendentes. Esta task consolida as correcoes necessarias antes de considerar a TASK-020 production-ready.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.gitignore`, `rust/tweet-preprocessor/README.md`, `benchmarks/preprocessing_benchmark.py`, `.claude/registry.md`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum (correcoes e documentacao)

#### Criterios de Aceite
- [x] .gitignore inclui `**/target/` para Rust build artifacts
- [x] README do Rust documenta divergencia de emoji handling (ou correcao implementada)
- [x] Benchmark suprime speedup quando parity check falhar
- [x] Benchmark valida row count antes de comparar outputs
- [x] Findings registrados nas Observacoes da TASK-020 no registry.md

#### Restricoes
- Priorizar documentacao de divergencia sobre correcao de emoji (complexidade alta para escopo desta task)
- Manter paridade funcional nos casos comuns (emojis single-codepoint)

#### Referencias
- Review adversarial Codex: 7 findings (2 HIGH, 4 MEDIUM, 1 LOW)
- TASK-020: CLI Rust para preprocessing

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-07 | 1 | Correcoes: .gitignore, README emoji docs, benchmark parity/seed/rowcount, registry | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-07
- **Branch:** feat/TASK-020-024-rust-preprocessing
- **Commit(s):** 6818885 fix(benchmark): add parity validation and document emoji divergence
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Findings HIGH corrigidos: (1) divergencia emoji documentada no README; (2) benchmark suprime speedup quando parity falha. Findings MEDIUM corrigidos: row count validation, seed reprodutibilidade. Findings aceitos: null handling (edge case raro), emoji loop allocation (premature optimization). PR #19.

---

### TASK-020
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-05-07

#### Objetivo
Implementar CLI em Rust para preprocessing de tweets em escala (1.6M+), com output em Parquet.

#### Contexto
O pipeline atual de preprocessing em Python (regex) e gargalo para processar 1.6M tweets. Rust oferece ganho de 10-20x em performance para operacoes de texto. A CLI recebe CSV/Parquet de entrada e produz Parquet limpo para consumo pelo pipeline de inferencia Python. Esta task pode ser executada em paralelo com TASK-007 (fine-tuning).

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `rust/tweet-preprocessor/` (novo projeto Cargo), `benchmarks/preprocessing_benchmark.py`
- **Dependencias necessarias:** Rust toolchain, crates: `clap` (CLI), `polars` (I/O), `regex`, `rayon` (paralelismo)
- **Impacto em funcionalidades existentes:** nenhum (componente novo e independente)

#### Criterios de Aceite
- [x] Projeto Cargo inicializado em `rust/tweet-preprocessor/`
- [x] CLI aceita `--input` (CSV/Parquet) e `--output` (Parquet)
- [x] Pipeline de limpeza equivalente ao Python: URLs, mentions, hashtags, emojis, lowercase
- [x] Processamento paralelo com rayon (usar todos os cores)
- [x] Benchmark documentado: tweets/segundo vs Python
- [x] README com instrucoes de build e uso

#### Restricoes
- Manter paridade funcional com `src/preprocessing.py` — output deve ser identico para os mesmos inputs
- Usar `polars` para I/O (nao `csv` crate puro) — melhor performance e compatibilidade com ecossistema Python

#### Referencias
- https://docs.rs/polars/latest/polars/
- https://docs.rs/rayon/latest/rayon/
- https://github.com/BurntSushi/ripgrep (referencia de CLI Rust performatica)

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-07 | 1 | Reconhecimento: codigo main.rs completo, Cargo.toml corrigido (edition 2024->2021), README criado, benchmark script criado. Rust toolchain nao instalado — build pendente. | em andamento |
| 2026-05-07 | 1 | Rust instalado, API polars 0.46 corrigida (removido JSON), build release concluido, benchmark executado: 42x speedup em 100k tweets | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-07
- **Branch:** feat/TASK-020-024-rust-preprocessing
- **Commit(s):** 1bdc807 feat(rust): add high-performance tweet preprocessing CLI
- **Avaliacao pos-implementacao:** aprovado com ressalvas
- **Observacoes:** Speedup medido: 2.1x (1k), 10.3x (10k), 42.2x (100k). Suporte JSON removido por incompatibilidade API polars 0.46. Build compilado em C:\temp\ por politica de seguranca OneDrive. Review Codex: 7 findings (corrigidos em TASK-024). PR #19.

---

### TASK-019
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-04

#### Objetivo
Atualizar README.md conforme regra 12-portfolio-publico.md: contexto de negocio, diagrama de arquitetura, decisoes de engenharia e instrucoes de execucao atualizadas.

#### Contexto
README atual esta desatualizado — nao reflete o estado real do projeto (training module, baseline, CI, testes). A regra 12 exige README com contexto de negocio, diagrama Mermaid, decisoes de engenharia e setup funcional.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `README.md`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Contexto de negocio explicando o problema real que o projeto resolve
- [x] Diagrama de arquitetura em Mermaid
- [x] Secao de decisoes de engenharia com justificativas
- [x] Secao Project Structure atualizada com todos os arquivos atuais
- [x] Secao Current Status atualizada refletindo estado real
- [x] README em ingles conforme regra 12.2

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-04 | 1 | Reescrita completa do README conforme regra 12 | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-04
- **Branch:** main
- **Commit(s):** pendente
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** README reescrito com 6 secoes da regra 12.2. Dados de baseline, decisoes tecnicas e estrutura verificados contra codebase real e registry.md.

---

### TASK-018
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** patch
- **Data de criacao:** 2026-05-04

#### Objetivo
Sincronizar regra faltante de .claude_config/ para .claude/, atualizar CLAUDE.md e remover pasta duplicada .claude_config/.

#### Contexto
Auditoria identificou que .claude_config/ contem regra 12-portfolio-publico.md ausente em .claude/rules/. Todos os demais 22 arquivos compartilhados sao identicos. A pasta .claude_config/ e uma copia redundante que deve ser removida apos sincronizacao.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.claude/rules/12-portfolio-publico.md` (novo), `.claude/CLAUDE.md`, `.claude_config/` (remocao)
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Arquivo 12-portfolio-publico.md presente em .claude/rules/
- [x] CLAUDE.md com referencia a regra 12 na estrutura do sistema de regras
- [x] Pasta .claude_config/ removida

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-04 | 1 | Auditoria, sincronizacao da regra 12, atualizacao CLAUDE.md, remocao .claude_config/ | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-04
- **Branch:** main
- **Commit(s):** pendente de commit pelo desenvolvedor
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** 22 arquivos compartilhados eram identicos. Unica diferenca acionavel era regra 12-portfolio-publico.md. registry.md de .claude_config/ era template vazio (descartado — .claude/ contem dados reais).

---

### TASK-017
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-01

#### Objetivo
Configurar pipeline CI com GitHub Actions para lint (ruff) e testes (pytest) em push/PR para main.

#### Contexto
Garantir que todo PR e push para main passe por lint e testes automatizados antes do merge. Pipeline minima sem Docker build (pertence a TASK-009/012). CI parcial antecipada da TASK-009.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.github/workflows/ci.yml`, `pyproject.toml`, `requirements-dev.txt`, `README.md`, `src/preprocessing.py`
- **Dependencias necessarias:** ruff, pytest (dev-only)
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Workflow CI com jobs lint e test configurado em .github/workflows/ci.yml
- [x] pyproject.toml com config ruff (rules E/F/I, line-length=120, py310) e pytest markers
- [x] requirements-dev.txt com ruff e pytest
- [x] ruff check . e ruff format --check . passam sem erros
- [x] pytest tests/ -m "not slow" -v passa
- [x] Badge de CI visivel no README.md

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-01 | 1 | Implementacao completa: ci.yml, pyproject.toml, requirements-dev.txt, badge, formatting fixes | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-01
- **Branch:** ci/TASK-017-github-actions-ci
- **Commit(s):** 69f970c style(preprocessing): fix formatting for ruff compliance, dd1dc67 ci(actions): add GitHub Actions pipeline with lint and tests
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Notebooks excluidos do ruff via extend-exclude. Import sorting corrigido em src/ e tests/. Testes clean_tweet corrigidos (expectativa [URL] -> [url] por conta do to_lowercase no pipeline).

---

### TASK-016
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-01

#### Objetivo
Integrar as diretrizes do repositorio andrej-karpathy-skills ao CLAUDE.md do projeto como secao obrigatoria.

#### Contexto
As diretrizes de Andrej Karpathy (forrestchang/andrej-karpathy-skills) codificam principios para reduzir erros comuns de LLMs ao gerar codigo. Os principios sao complementares as regras existentes em .claude/rules/ e devem ser incorporados como referencia obrigatoria.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.claude/CLAUDE.md`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Secao com as diretrizes Karpathy adicionada ao CLAUDE.md
- [x] Referencia ao repositorio original incluida
- [x] Secao marcada como obrigatoria para agentes de IA

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-01 | 1 | Diretrizes integradas ao CLAUDE.md com correspondencia as regras existentes | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-01
- **Branch:** docs/TASK-016-karpathy-skills
- **Commit(s):** e80b163 docs(claude): add Karpathy behavioral guidelines as mandatory section
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Os 4 principios Karpathy ja estavam codificados em rules/01-principios.md. A integracao adicionou a secao ao CLAUDE.md com correspondencia explicita, marcada como obrigatoria e com referencia ao repositorio original.

---

### TASK-015
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-05-01

#### Objetivo
Corrigir hooks pre-commit e pre-push para implementar validacoes faltantes da regra 09.1.

#### Contexto
Auditoria pos-TASK-000 identificou dois desvios: (1) pre-commit nao valida se arquivos staged estao no escopo da task ativa; (2) pre-push nao verifica se tasks concluidas referenciadas nos commits possuem entrada no registry.md.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.claude/hooks/pre-commit`, `.claude/hooks/pre-push`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** hooks existentes estendidos, comportamento anterior preservado

#### Criterios de Aceite
- [x] pre-commit valida se arquivos staged estao no Escopo Tecnico da task ativa em tasks.md
- [x] pre-push valida se tasks concluidas nos commits possuem entrada no historico do registry.md
- [x] Ambas as validacoes emitem warning (nao-bloqueante) em caso de duvida

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-01 | 1 | Implementacao das validacoes de escopo e registry | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-01
- **Branch:** fix/TASK-015-corrigir-hooks-enforcement
- **Commit(s):** 4300b61 fix(enforcement): add scope and registry validation to pre-commit and pre-push hooks
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Validacao de escopo compara staged files contra campo Escopo Tecnico da task ativa. Validacao de registry extrai TASK-NNN dos commits e verifica presenca no historico. Arquivos .claude/ sao sempre permitidos no escopo.

---

### TASK-000
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-05-01

#### Objetivo
Instalar hooks de enforcement git, enforcement.conf e templates de PR/Issue conforme regra 09.

#### Contexto
Bootstrap obrigatorio do sistema de governanca. Os hooks validam automaticamente formato de commits, debug statements, branches e estado do registro.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.claude/hooks/commit-msg`, `.claude/hooks/pre-commit`, `.claude/hooks/pre-push`, `.claude/hooks/post-merge`, `.claude/enforcement.conf`, `.claude/pr-template.md`, `.claude/issue-template.md`
- **Dependencias necessarias:** nenhuma (bash + git puro)
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Hook commit-msg valida formato type(scope): subject, sem body, sem co-authored-by
- [x] Hook pre-commit verifica debug statements nos arquivos staged
- [x] Hook pre-push valida formato da branch e task ativa
- [x] Hook post-merge emite aviso de verificacao pos-pull
- [x] enforcement.conf com patterns de debug por linguagem
- [x] pr-template.md e issue-template.md criados
- [x] git config core.hooksPath apontando para .claude/hooks

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-01 | 1 | Registro da task e implementacao completa | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-01
- **Branch:** dev
- **Commit(s):** 014255f feat(enforcement): add git hooks, enforcement.conf and PR/issue templates
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** 4 hooks (commit-msg, pre-commit, pre-push, post-merge), enforcement.conf e templates criados. git config core.hooksPath configurado.

---

### TASK-001
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-03-10

#### Objetivo
Criar repositorio Git, estrutura de pastas, ambiente virtual e requirements.txt com dependencias fixas.

#### Contexto
Primeira task do projeto — pre-requisito para todas as demais. Setup inicial com repositorio GitHub, .gitignore e dependencias instaladas.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `requirements.txt`, `.gitignore`, estrutura de diretorios base
- **Dependencias necessarias:** Python 3.10+, pip
- **Impacto em funcionalidades existentes:** nenhum (projeto novo)

#### Criterios de Aceite
- [x] Repositorio no GitHub criado e acessivel
- [x] Estrutura de pastas do projeto definida
- [x] Dependencias instalaveis via pip install -r requirements.txt

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-03-10 | 1 | Setup inicial concluido | concluida |

#### Resultado
- **Data de conclusao:** 2026-03-10
- **Branch:** dev
- **Commit(s):** cc2f785 chore: init project structure with base dependencies
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** —

---

### TASK-002
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-03-09

#### Objetivo
Realizar analise exploratoria do dataset tweet_eval (sentiment) com distribuicao de classes, comprimento de tweets e padroes de ruido.

#### Contexto
EDA necessaria para entender o dataset antes de implementar preprocessing e treinamento. Passou por revisao com 9 itens de correcao identificados e endereados antes da aprovacao.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `notebooks/01_eda.ipynb`
- **Dependencias necessarias:** nenhuma (datasets, matplotlib, pandas ja instaladas)
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] 4 perguntas analiticas respondidas com dados e visualizacoes
- [x] Subplot unificado de distribuicao de classes (1x3) e histogramas (2x3)
- [x] Variaveis nomeadas conforme padrao VAR (train_data, val_data, test_data)
- [x] Analise de padroes de ruido via regex incluida

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-03-09 | 1 | Primeira entrega — code review reprovado, 9 itens de correcao | em andamento |
| 2026-03-19 | 2 | Resubmissao — todos os 9 itens endereados | concluida |

#### Resultado
- **Data de conclusao:** 2026-03-19
- **Branch:** dev
- **Commit(s):** cfc1909 feat(data): add exploratory data analysis notebook
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Revisao identificou 9 itens de correcao na primeira entrega

---

### TASK-003
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-03-20

#### Objetivo
Construir pipeline de pre-processamento especifico para tweets com remocao de URLs, mencoes, normalizacao de hashtags, conversao de emojis e lowercase.

#### Contexto
Pipeline prepara texto limpo para o tokenizador do modelo RoBERTa. Decisoes: URLs substituidas por token [URL], emojis convertidos via emoji.demojize().

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `src/preprocessing.py`
- **Dependencias necessarias:** emoji (ja instalada)
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Funcoes de limpeza implementadas (remove_urls, remove_mentions, normalize_hashtags, handle_emojis, to_lowercase)
- [x] Funcao clean_tweet_text aplica pipeline completo
- [x] Regex patterns pre-compilados no nivel do modulo

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-03-21 | 1 | Pipeline implementado e commitado | concluida |

#### Resultado
- **Data de conclusao:** 2026-03-21
- **Branch:** dev
- **Commit(s):** 65d9e4b feat(preprocessing): add tweet cleaning pipeline, 4e2cc37 fix(preprocessing): replace url removal with [URL] token, b224044 refactor(preprocessing): precompile regex patterns at module level
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Refatoracao posterior para pre-compilar regex patterns

---

### TASK-004
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-03-20

#### Objetivo
Implementar testes unitarios para o pipeline de pre-processamento cobrindo casos principais e extremos.

#### Contexto
Cobertura de testes para todas as funcoes de limpeza da TASK-003, incluindo casos normais e extremos.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `tests/test_preprocessing.py`
- **Dependencias necessarias:** pytest (ja instalada)
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Testes para remove_urls: tweet com URL e tweet sem URL
- [x] Testes para remove_mentions e demais funcoes
- [x] 12 testes passando via pytest
- [x] Cobertura de casos extremos

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-03-21 | 1 | 12 testes implementados e passando | concluida |

#### Resultado
- **Data de conclusao:** 2026-03-21
- **Branch:** dev
- **Commit(s):** 556844f test(preprocessing): add unit tests for cleaning functions
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** —

---

### TASK-005
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-03-24

#### Objetivo
Analisar distribuicao de comprimento de tokens por split e validar max_length=128 para o fine-tuning.

#### Contexto
Tokenizador cardiffnlp/twitter-roberta-base-sentiment aplicado sobre o dataset para determinar o limite de truncamento adequado. Percentil 99 ficou em ~55 tokens, validando max_length=128 como conservador.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `notebooks/02_tokenization.ipynb`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Tokenizador carregado e aplicado sobre o dataset completo
- [x] Distribuicao de comprimento de tokens plotada por split
- [x] Percentual de truncamento calculado e documentado

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-03-26 | 1 | Analise concluida, max_length=128 validado | concluida |

#### Resultado
- **Data de conclusao:** 2026-03-26
- **Branch:** dev
- **Commit(s):** 347338d feat(tokenization): add token count distribution and define max_length
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** max_length=128 cobre 99%+ das amostras; max_length=64 seria seguro mas optou-se pelo conservador

---

### TASK-006
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** minor
- **Data de criacao:** 2026-03-28

#### Objetivo
Executar inferencia zero-shot com o modelo pre-treinado sobre o test split completo e estabelecer baseline oficial de desempenho.

#### Contexto
Baseline quantitativo para avaliar o ganho real do fine-tuning. Modelo cardiffnlp/twitter-roberta-base-sentiment sem fine-tuning aplicado sobre os 12.284 exemplos de teste.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `notebooks/03_inference_baseline.ipynb`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum

#### Criterios de Aceite
- [x] Pipeline de inferencia executado sobre o test split completo
- [x] Relatorio de metricas (accuracy, macro F1, precision e recall por classe)
- [x] Matriz de confusao plotada
- [x] Resultados documentados como baseline oficial

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-03-30 | 1 | Baseline executado: 70% acc, 0.71 F1 macro | concluida |

#### Resultado
- **Data de conclusao:** 2026-03-30
- **Branch:** dev
- **Commit(s):** 409684c feat(eval): add zero-shot baseline inference and metrics report
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Baseline: accuracy 70%, F1 macro 0.71. Negative 0.70, Neutral 0.70, Positive 0.73 (F1 por classe)

---

### TASK-014
- **Status:** concluida
- **Modo:** desenvolvimento
- **Complexidade:** major
- **Data de criacao:** 2026-05-01

#### Objetivo
Migrar tasks do arquivo tasks_para_mapear para o formato .claude em tasks.md e atualizar registry.md e CLAUDE.md com dados reais do projeto.

#### Contexto
O projeto possui 13 tasks documentadas em formato proprio no arquivo tasks_para_mapear, mas o sistema de governanca .claude estava no estado de bootstrap — tasks.md vazio, registry.md com placeholders, CLAUDE.md com dados genericos. Necessario alinhar toda a documentacao ao formato mandatorio.

#### Escopo Tecnico
- **Arquivos/modulos envolvidos:** `.claude/tasks.md`, `.claude/registry.md`, `.claude/CLAUDE.md`
- **Dependencias necessarias:** nenhuma
- **Impacto em funcionalidades existentes:** nenhum (apenas documentacao de governanca)

#### Criterios de Aceite
- [x] 13 tasks migradas para o formato template de tasks.md
- [x] CLAUDE.md com dados reais do projeto (nome, stack, repositorio, comandos)
- [x] registry.md com historico de implementacoes e estado atualizado
- [x] Tasks concluidas (1-6) com Resultado preenchido retroativamente
- [x] Tasks ativas (7-13) com campos obrigatorios

#### Log de Andamento

| Data | Sessao | Acao Realizada | Status ao Final |
|------|--------|----------------|-----------------|
| 2026-05-01 | 1 | Analise comparativa tasks_para_mapear vs formato .claude | em andamento |
| 2026-05-01 | 1 | Migracao completa: CLAUDE.md, tasks.md, registry.md | concluida |

#### Resultado
- **Data de conclusao:** 2026-05-01
- **Branch:** dev
- **Commit(s):** 73215cc docs(claude): migrate tasks and update project registry
- **Avaliacao pos-implementacao:** aprovado
- **Observacoes:** Campos nao mapeados no formato .claude foram ignorados conforme instrucao (Sprint, DoD, Estrategia de Teste, Dependencias e Bloqueios)

---

## Tasks Descartadas

> Tasks que foram canceladas ou substituidas antes da implementacao. Registre o motivo.

[nenhuma task descartada]

## Regras de Preenchimento

1. **O campo Objetivo deve caber em uma frase.** Se nao cabe, a task e grande demais — quebre em subtasks.
2. **Uma task deve ser completavel em uma sessao de desenvolvimento.** Se a estimativa de implementacao excede uma sessao, ou se a task afeta mais de 10 arquivos, ela deve ser decomposta em subtasks independentes. Cada subtask recebe seu proprio TASK-NNN e segue o fluxo completo. O campo Contexto da subtask deve referenciar a task mae.
3. **Criterios de Aceite sao obrigatorios e verificaveis.** "Funcionar corretamente" nao e criterio. "Retornar status 200 para inputs validos e 400 para inputs invalidos" e.
4. **Escopo Tecnico deve listar arquivos concretos.** "Algumas telas" nao serve. "src/screens/LoginScreen.tsx, src/services/authService.ts" serve.
5. **Uma task por implementacao.** Se durante o desenvolvimento surgir necessidade de outra mudanca fora do escopo, registre uma nova task — nao expanda a atual.
6. **Tasks nao sao retroativas.** Codigo ja implementado sem task registrada deve ser revisado (Modo Review) e documentado antes de prosseguir com novas tasks.
7. **O resultado e preenchido pelo agente** ao final da implementacao, junto com a atualizacao do Registro de Projeto.
8. **Complexidade e obrigatoria.** Toda task deve ser classificada como `patch`, `minor` ou `major`. Na duvida, classifique para cima (minor em vez de patch, major em vez de minor). A classificacao determina o nivel de cerimonia da avaliacao pos-implementacao.
9. **A ordem na secao Tasks Ativas define prioridade.** A primeira task e a ativa. O agente nao pula para a segunda sem que a primeira esteja concluida, descartada ou explicitamente pausada pelo usuario.
10. **O Log de Andamento e obrigatorio para tasks `minor` e `major`.** O agente registra uma entrada a cada sessao em que trabalhar na task, incluindo interrupcoes e travamentos. Tasks `patch` podem omitir o log. O log captura o progresso intermediario; a conclusao final e registrada no Resultado da task e no Historico de Implementacoes do `registry.md`.
11. **Tasks revertidas nao sao deletadas.** Ao reverter uma implementacao, a task original recebe status `revertida` com nota explicativa, e uma nova task `fix` ou `revert` e criada referenciando a original.
