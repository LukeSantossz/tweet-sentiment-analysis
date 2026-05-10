# Registro de Projeto — Estado e Historico

> Este arquivo contem o estado atual e historico do projeto. E atualizado pelo agente ao final de cada implementacao.
> As **regras** sobre como atualizar este registro estao em `.claude/rules/08-registro-projeto.md`.

---

## Informacoes do Projeto

- **Nome:** tweet-sentiment-analysis
- **Stack:** Python 3.10 · HuggingFace Transformers · RoBERTa · scikit-learn · pytest
- **Repositorio:** LukeSantossz/tweet-sentiment-analysis
- **Estrutura:** src/ (modulos Python), notebooks/ (analises Jupyter), tests/ (pytest), data/ (datasets), outputs/ (checkpoints)

## Historico de Implementacoes

> Registro de conclusoes. Cada entrada representa uma task finalizada — nao o progresso intermediario (que vive no Log de Andamento de cada task em `tasks.md`). O agente adiciona uma nova linha apos cada task concluida. Nunca remova entradas anteriores.

| # | Data | Task | Complexidade | Escopo Alterado | Resultado | Observacoes |
|---|------|------|--------------|-----------------|-----------|-------------|
| 1 | 2026-03-10 | TASK-001 | minor | 3 arquivos — config e estrutura | aprovado | Setup inicial do repositorio |
| 2 | 2026-03-19 | TASK-002 | minor | 1 arquivo — notebooks | aprovado | EDA com 9 correcoes na revisao |
| 3 | 2026-03-21 | TASK-003 | minor | 1 arquivo — src/preprocessing | aprovado | Pipeline de limpeza de tweets |
| 4 | 2026-03-21 | TASK-004 | minor | 1 arquivo — tests | aprovado | Testes unitarios do preprocessing |
| 5 | 2026-03-26 | TASK-005 | minor | 1 arquivo — notebooks | aprovado | Analise de tokenizacao, max_length=128 |
| 6 | 2026-03-30 | TASK-006 | minor | 1 arquivo — notebooks | aprovado | Baseline zero-shot: 70% acc, 0.71 F1 |
| 7 | 2026-05-01 | TASK-014 | major | 3 arquivos — .claude/ | aprovado | Migracao de tasks para formato .claude |
| 8 | 2026-05-01 | TASK-000 | major | 7 arquivos — .claude/hooks/, enforcement.conf, templates | aprovado | Bootstrap: hooks, enforcement.conf, templates |
| 9 | 2026-05-01 | TASK-015 | minor | 2 arquivos — .claude/hooks/ | aprovado | Correcao: validacao de escopo e registry nos hooks |
| 10 | 2026-05-01 | TASK-016 | minor | 1 arquivo — .claude/CLAUDE.md | aprovado | Diretrizes Karpathy integradas como secao obrigatoria |
| 11 | 2026-05-01 | TASK-017 | minor | 7 arquivos — CI, config, src/, tests/ | aprovado | GitHub Actions CI com lint (ruff) e testes (pytest) |
| 12 | 2026-05-04 | TASK-018 | patch | 2 arquivos + remocao .claude_config/ | aprovado | Sincronizacao regra 12 + remocao pasta duplicada |
| 13 | 2026-05-04 | TASK-019 | minor | 1 arquivo — README.md | aprovado | README reescrito conforme regra 12-portfolio-publico |
| 14 | 2026-05-07 | TASK-020 | major | 4 arquivos — rust/tweet-preprocessor/, benchmarks/ | aprovado com ressalvas | CLI Rust: 42x speedup. Review Codex: 7 findings (2 HIGH corrigidos em TASK-024) |
| 15 | 2026-05-07 | TASK-024 | minor | 4 arquivos — .gitignore, rust/README, benchmark, registry | aprovado | Correcoes do review Codex: emoji docs, benchmark parity/seed |
| 16 | 2026-05-09 | TASK-025 | minor | 4 arquivos — requirements.txt, main.rs, Cargo.toml, tasks.md | aprovado | Auditoria: emoji multi-codepoint, deps, tasks reorganizadas |

## Estado da Codebase

> Atualizado a cada implementacao ou verificacao pos-pull. Reflete o snapshot mais recente do projeto.

- **Ultima atualizacao:** 2026-05-09
- **Ultimo responsavel:** agente
- **Branch ativa:** feat/TASK-020-024-rust-preprocessing
- **Dependencias alteradas recentemente:** polars e numpy adicionados ao requirements.txt, unicode-segmentation adicionado ao Cargo.toml
- **Testes passando:** sim — 20 testes Python (12 preprocessing + 8 training) + 7 testes Rust
- **Divergencias externas pendentes:** nenhuma
- **Ultima task concluida:** TASK-025 — Correcoes de auditoria (emoji multi-codepoint, requirements, tasks reorganizadas)

## Pendencias Conhecidas

- TASK-007 em andamento: script de fine-tuning implementado (src/training.py + tests), execucao do treinamento pendente (requer GPU)
- outputs/ vazio: nenhum checkpoint de modelo gerado ainda
- preprocessing.py nao e usado pelo pipeline de training (training.py carrega direto do HF Hub)
- Pipeline de escala: TASK-020 concluida (Rust CLI), TASK-021 a TASK-023 pendentes (batch inference, benchmark, docs)

## Decisoes Tecnicas Relevantes

> Decisoes tomadas durante implementacoes que afetam futuras tasks. Inclua justificativa breve.

- Modelo base: cardiffnlp/twitter-roberta-base-sentiment (pre-treinado em tweets, alinhado ao dominio)
- max_length=128 tokens (percentil 99 ~55 tokens, margem conservadora)
- Metrica principal: F1 macro (dataset desbalanceado — neutral ~45-50%, positive ~30-35%, negative ~20-25%)
- URLs substituidas por token [URL] no preprocessing (preserva informacao de presenca sem ruido)
- Emojis convertidos via emoji.demojize() (transforma em texto descritivo legivel pelo tokenizador)
- Early stopping com patience=2 no training (evitar overfitting)
- Ruff: rules E/F/I, line-length=120, notebooks excluidos via extend-exclude (nao sao codigo de producao)
- CI: torch CPU-only no pipeline para evitar download CUDA (~2GB); testes slow excluidos via marker
- Arquitetura de escala: Rust CLI para preprocessing (42x speedup medido em 100k tweets) + Python para inferencia GPU (minimo 1M tweets)
- Polars 0.46 Rust: suporte JSON removido por incompatibilidade de API; CSV e Parquet suportados
- Emoji handling Rust: corrigido para usar grapheme clusters (unicode-segmentation crate) em vez de char iteration; paridade total com Python para emojis multi-codepoint (flags, skin tones, ZWJ sequences)

## Padroes Recorrentes Observados

| Padrao | Frequencia | Impacto | Acao Corretiva |
|--------|------------|---------|----------------|
| [nenhum registrado] | — | — | — |

---

## Notas de Sessao

> Espaco para anotacoes pontuais sobre contextos que influenciam futuras sessoes.

- [2026-05-01] Migracao retroativa: tasks 1-6 foram implementadas antes do sistema .claude estar operacional. Documentacao reconstruida a partir do git log e do arquivo tasks_para_mapear.
- [2026-05-07] Review adversarial Codex da TASK-020 identificou 7 findings: (HIGH) paridade emoji multi-codepoint — documentado; (HIGH) benchmark reporta speedup sem parity — corrigido; (MEDIUM) row count validation — corrigido; (MEDIUM) benchmark seed — corrigido; (MEDIUM) null handling — aceito; (MEDIUM) emoji loop allocation — aceito (premature optimization); (LOW) README JSON — corrigido.
