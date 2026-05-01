# tweet-sentiment-analysis — Diretrizes para Agentes de IA

> **Este projeto opera sob um fluxo mandatório.** Nenhum agente de IA pode modificar a codebase sem task registrada em `.claude/tasks.md`. Consulte `.claude/rules/` para as regras completas.

## Projeto

- **Nome:** tweet-sentiment-analysis
- **Stack:** Python 3.10 · HuggingFace Transformers · RoBERTa · scikit-learn · pytest
- **Repositório:** LukeSantossz/tweet-sentiment-analysis
- **Estrutura:** src/ (módulos Python), notebooks/ (análises Jupyter), tests/ (pytest), data/ (datasets), outputs/ (checkpoints)

## Comandos

```bash
# Testes
pytest tests/

# Lint
# não configurado (previsto na TASK-009)

# Type check
# não aplicável

# Rodar aplicação
python -m src.training

# Docker (se aplicável)
# não configurado (previsto na TASK-012)
```

## Estrutura do Sistema de Regras

```
projeto/
├── CLAUDE.md                          ← este arquivo (entrada do projeto)
├── .claude/
│   ├── rules/
│   │   ├── 00-trava-seguranca.md      ← condições obrigatórias de operação
│   │   ├── 01-principios.md           ← pense antes de codar, simplicidade, cirúrgico
│   │   ├── 02-reconhecimento.md       ← inventário técnico pré-implementação
│   │   ├── 03-modos-operacao.md       ← desenvolvimento, review, tutor
│   │   ├── 04-avaliacao-pos.md        ← protocolo pós-implementação
│   │   ├── 05-convencoes.md           ← VAR Method, Conventional Commits, branches
│   │   ├── 06-crura.md               ← fluxo CRURA + checklist + reversão + templates
│   │   ├── 07-integridade.md          ← 12 regras invioláveis
│   │   ├── 08-registro-projeto.md     ← regras de atualização do registry
│   │   ├── 09-enforcement.md          ← hooks git automatizados
│   │   ├── 10-engenharia-agentica.md  ← metodologia Karpathy, checklist agêntico
│   │   └── 11-integracao-codex.md     ← orquestração dual-agent com Codex
│   ├── registry.md                    ← estado do projeto + histórico (mutável)
│   ├── registry-archive.md            ← criado automaticamente quando histórico > 30 entradas
│   ├── tasks.md                       ← registro de tasks (obrigatório)
│   ├── pr-template.md                 ← template de Pull Request
│   ├── issue-template.md              ← template de Issue
│   ├── guia-configuracao-codex.md     ← guia prático de integração com Codex
│   ├── setup-hooks.sh                 ← script de configuração dos git hooks
│   ├── hooks/                         ← scripts de enforcement git
│   └── enforcement.conf               ← padrões de debug log por linguagem
```

## Fluxo Resumido

1. **Task registrada** em `tasks.md` → obrigatório antes de qualquer código
2. **Modo declarado** (Desenvolvimento / Review / Tutor)
3. **Reconhecimento** da codebase
4. **Implementação** seguindo princípios e convenções
5. **Avaliação pós-implementação** (automática pelo agente)
6. **Atualização** do `registry.md`
7. **CRURA** — Change → Review → Upload → Review Again → Auto-Revisão

## Diretrizes Karpathy (obrigatório)

> Baseado em [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills).
> Estas diretrizes sao obrigatorias para todo agente de IA que opere neste projeto.
> Bias: cautela sobre velocidade. Para tasks triviais, use julgamento.

### 1. Pense Antes de Codar
Nao assuma. Nao esconda confusao. Exponha trade-offs. Se houver multiplas interpretacoes, apresente-as — nao escolha silenciosamente. Se algo estiver confuso, pare, nomeie o que esta confuso e pergunte. Correspondencia: `rules/01-principios.md` secao 1.1.

### 2. Simplicidade Primeiro
Codigo minimo que resolve o problema. Nada especulativo. Sem features alem do pedido, sem abstracoes para uso unico, sem flexibilidade nao solicitada, sem tratamento de erro para cenarios impossiveis. Se escreveu 200 linhas e 50 resolveriam, reescreva. Teste: "Um engenheiro senior diria que isso esta overengineered?" Se sim, simplifique. Correspondencia: `rules/01-principios.md` secao 1.2.

### 3. Mudancas Cirurgicas
Toque apenas no necessario. Nao "melhore" codigo adjacente, comentarios ou formatacao. Nao refatore o que nao esta quebrado. Siga o estilo existente. Se notar codigo morto nao relacionado, mencione — nao delete. Remova orfaos criados pelas SUAS mudancas, nao codigo morto pre-existente. Teste: toda linha alterada deve ter rastreabilidade direta a solicitacao do usuario. Correspondencia: `rules/01-principios.md` secao 1.3.

### 4. Execucao Orientada a Objetivos
Defina criterios de sucesso verificaveis. Transforme tasks em objetivos concretos: "Adicionar validacao" se torna "Escrever testes para inputs invalidos, depois faze-los passar". Para multi-step, declare plano com checkpoints. Correspondencia: `rules/01-principios.md` secao 1.4.

**Estas diretrizes estao funcionando quando:** diffs contem menos mudancas desnecessarias, reescritas por overengineering diminuem, e perguntas de esclarecimento acontecem antes da implementacao.

## Convencoes Rapidas

- **Commits:** `type(scope): subject` — sem body, sem co-authored-by
- **Branches:** `type/TASK-NNN-descricao-curta`
- **Tasks:** uma por implementacao, complexidade obrigatoria (patch/minor/major)
- **Nomenclatura:** VAR Method (Data, Info, Manager, Handler, Service, Repository...)
