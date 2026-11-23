from pathlib import Path
from streamlit.testing.v1 import AppTest
 
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
 
 
def test_app_renders_with_defaults():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "Cache & bandwidth partitioning"
 
 
def test_app_accepts_manual_program_types():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    at.sidebar.checkbox[0].set_value(False).run(timeout=30)
    assert not at.exception
    assert len(at.sidebar.selectbox) == 4  # default n = 4
 
 
def test_app_changes_with_n():
    at = AppTest.from_file(str(APP)).run(timeout=30)
    at.sidebar.number_input[0].set_value(2).run(timeout=30)
    assert not at.exception
 