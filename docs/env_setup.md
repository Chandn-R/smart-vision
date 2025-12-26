# Python Environment Setup Guide

This guide will walk you step-by-step through:

* Installing Python 3.11
* Creating a virtual environment
* Installing dependencies from `environment/requirements.txt`
* Running the ML pipeline via `pipeline/run_inference.py`

---

## 1. **Verify Python 3.11 Installation**

Make sure Python 3.11 is installed on your system.

```bash
python3.11 --version
```

If it shows something like `Python 3.11.x`, you're good.

If not, download and install Python 3.11 from the official site:

* [https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/)

---

## 2. **Create a Virtual Environment (Python 3.11)**

Navigate to your project folder:

```bash
cd /path/to/your/project
```

Create a new environment:

```bash
python3.11 -m venv env
```

Activate the environment:

* **Mac/Linux:**

  ```bash
  source env/bin/activate
  ```
* **Windows:**

  ```bash
  env\Scripts\activate
  ```

You should now see `(env)` before your terminal prompt.

---

## 3. **Install Requirements from environment/requirements.txt**

Run the following command from the root of your project:

```bash
pip install -r environment/requirements.txt
```

This will install all required packages for your ML pipeline.

---

## 4. **Run the ML Pipeline**

Navigate to the pipeline directory:

```bash
cd pipeline
```

Run the pipeline script:

```bash
python run_inference.py
```

This will start your machine learning inference pipeline.

---

## 5. **Deactivate the Environment (Optional)**

When you're done:

```bash
deactivate
```
