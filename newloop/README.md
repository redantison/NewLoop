# NewLoop Streamlit App

NewLoop is a simulation app built with Streamlit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Set the main file path to `app.py`.

## Files

- `app.py`: Streamlit wrapper entrypoint
- `requirements.txt`: Python dependencies and editable package install
- `newloop/slnewloop.py`: Streamlit app module
- Core model modules: `newloop/engine.py`, `newloop/results.py`, and related files
