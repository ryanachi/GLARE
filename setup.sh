MODEL_SOURCE="sto"

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
pip install -e ./ilm
> .env
export ILM_DIR="$PWD"
echo "ILM_DIR=\"$PWD\"" >> .env
python ./ilm/acl20_repro.py model $MODEL_SOURCE ilm | bash
echo "MODEL_DIR=\"$ILM_DIR/models/${MODEL_SOURCE}_ilm\"" >> .env