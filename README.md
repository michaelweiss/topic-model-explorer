# Topic Model Explorer

This is a tool for exploring topic models built on top of [streamlit.io](https://www.streamlit.io). There are two versions of the tool with different objectives:

## tme.py

Supports the workflow to create topic co-occurrence networks and keyword co-occurrence networks. To use this tool, run:

```
streamlit run tme.py
```

## tme-s.py 

Creates a synthesis of multiple runs of a topic model. This is important when using short documents, as there can be quite some variation between individual runs.

```
streamlit run tme-s.py
```