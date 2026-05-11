# CLAUDE.md — Ponto de Entrada do Framework de Desenvolvimento

> **Versão:** 1.1.0 | **Localização das regras:** `.claude/rules/` | **Estado:** `.claude/tasks.md` + `.claude/registry.md`

---

## Trava de Segurança (Regra 00 — Incondicional)

Nenhuma implementação, modificação, criação ou exclusão de código é permitida sem:

1. **Task registrada** em `.claude/tasks.md`
2. **Modo declarado** pelo usuário (Desenvolvimento, Review ou Tutor)
3. **Codebase reconhecida** (regra 02 executada)
4. **Registry verificado** (`.claude/registry.md` lido)

Exceções: Modo Tutor e Review podem iniciar sem task, mas qualquer modificação de código exige registro prévio. Detalhes completos: `.claude/rules/00-trava-seguranca.md`.

## Princípios Core (Regra 01)

- Pense antes de codar. Declare premissas, exponha trade-offs, pergunte se ambíguo.
- Simplicidade primeiro. Código mínimo, sem features especulativas, sem abstração prematura.
- Mudanças cirúrgicas. Toque apenas o necessário. Limpe apenas a própria sujeira.
- Todo código gerado por agente é rascunho até ser revisado e compreendido pelo desenvolvedor.

## Início de Sessão — O Que Ler

### Sempre (toda sessão):

1. Este arquivo (`CLAUDE.md`)
2. `.claude/registry.md` → estado atual, última implementação, pendências
3. `.claude/tasks.md` → **apenas a seção "Tasks Ativas"**, não carregar Tasks Concluídas

### Sob demanda (quando a condição ativar):

| Condição | Ler |
|----------|-----|
| Projeto novo ou primeira sessão | `.claude/prd.md` (se existir) |
| Task `minor` ou `major` | Regras 04 (avaliação) + 06 (CRURA) + 08 (registro) |
| Task `patch` | Apenas regra 05 (convenções) para commit |
| Modo Review ativado | Regra 03 completa (protocolo de review) |
| Modo Tutor ativado | Regra 03 completa (método de dicas progressivas) |
| Publicar no GitHub / curar portfólio | `.claude/guides/guia-portfolio.md` |
| Usar integração Codex | `.claude/guides/guia-codex.md` |
| Setup de hooks ou enforcement | Regra 09 |
| Dúvida sobre nomenclatura ou commits | Regra 05 |
| Task requer referência a padrões anteriores | Consultar base de conhecimento externa (ver seção abaixo) |

### Regras detalhadas (referência completa):

```
.claude/rules/
├── 00-trava-seguranca.md     ← condições obrigatórias
├── 01-principios.md          ← como pensar e codar
├── 02-reconhecimento.md      ← mapeamento pré-implementação
├── 03-modos-operacao.md      ← desenvolvimento / review / tutor
├── 04-avaliacao-pos.md       ← verificação pós-implementação + testes
├── 05-convencoes.md          ← nomenclatura, commits, branches
├── 06-crura.md               ← fluxo CRURA + checklist unificado
├── 07-integridade.md         ← regras invioláveis
├── 08-registro-projeto.md    ← registry + recuperação de sessão
└── 09-enforcement.md         ← hooks git automatizados
```

## Recuperação de Sessão

Se a sessão anterior foi interrompida (timeout, limite de contexto, crash):

1. Ler `registry.md` → última implementação e estado registrado
2. Ler `tasks.md` → task ativa e último Log de Andamento
3. Verificar branch atual (`git branch --show-current`) e último commit (`git log -1 --oneline`)
4. Comparar estado real vs registrado. Reportar divergências ao usuário.
5. Retomar do ponto documentado no Log de Andamento.

## Base de Conhecimento Externa

Caminho: C:\Users\lucas\OneDrive\Desktop\llm-wiki\wiki\
Índice: wiki/index.md

**Regras de uso:**
- APENAS CONSULTA — não modificar, criar ou atualizar arquivos nesta pasta
- Consultar antes de: decidir stack, investigar bugs recorrentes, tomar decisões arquiteturais
- O índice `index.md` é o ponto de entrada para navegação

## Informações do Projeto

- **Nome:** tweet-sentiment-analysis
- **Stack:** Python 3.10 · HuggingFace Transformers · RoBERTa · scikit-learn · pytest · Rust (tweet-preprocessor CLI)
- **Repositório:** LukeSantossz/tweet-sentiment-analysis
- **Estrutura:** src/ (módulos Python), notebooks/ (análises Jupyter), tests/ (pytest), data/ (datasets), outputs/ (checkpoints), rust/ (CLI de preprocessing)

## Comandos

```bash
# Testes Python
pytest tests/

# Testes Rust
cd rust/tweet-preprocessor && cargo test

# Lint
ruff check .

# Rodar training
python -m src.training

# Preprocessing Rust (CSV)
cd rust/tweet-preprocessor && cargo run --release -- -i data/input.csv -o data/output.csv -f csv

# Build Rust release
cd rust/tweet-preprocessor && cargo build --release
```

## Convenções Rápidas

- **Commits:** `type(scope): subject` — sem body, sem co-authored-by
- **Branches:** `type/TASK-NNN-descricao-curta`
- **Tasks:** uma por implementação, complexidade obrigatória (patch/minor/major)
- **Nomenclatura:** VAR Method (Data, Info, Manager, Handler, Service, Repository...)
