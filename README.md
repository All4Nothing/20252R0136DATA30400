# TELEClass Pipeline

## Prerequisites

- Python 3.8+
- PyTorch, transformers, sentence-transformers, pandas, networkx, scikit-learn, tqdm, python-dotenv, openai

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **Create `.env` file in the project root:**
```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

**⚠️ Important**: The `.env` file with `OPENAI_API_KEY` is **required** for the pipeline to work.

## LLM Configuration

- **Prompts**: LLM에 사용된 프롬프트는 `src/llm_outputs/prompt.txt`에 저장되어 있습니다.
- **LLM Outputs**: LLM 호출 결과는 `src/llm_outputs/` 디렉토리에 별도로 저장됩니다.

## Usage

### Run Complete Pipeline

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will:
1. Train the model
2. Run inference
3. Generate `src/outputs/submission.csv`

### Run Manually

```bash
cd src
python train_model.py
python inference_model.py
```

## Output

- Trained Model: `src/outputs/models/best_model/`
- Submission: `src/outputs/submission.csv`
