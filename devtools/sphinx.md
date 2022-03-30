
```
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs
# separate source and build = yes

cd docs

sphinx-apidoc -f -o source/ ../graphenv/
# add `modules` to toctree in index.rst
# make any needed changes to source/conf.py
make html
# now browse build/html/index.html

# to update, repeat the sphinx-apidoc and make html

```

