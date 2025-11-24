# dashboard

cd ~/quant-dashboard-finance

git pull origin main

source venv/bin/activate

pip install -r requirements.txt

deactivate

sudo systemctl restart streamlit-dashboard

sudo systemctl status streamlit-dashboard
