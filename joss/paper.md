---
title: 'graphenv: a Python library for reinforcement learning on graph search spaces'
tags:
  - Python
  - reinforcement learning
  - graph search
  - combinatorial optimization
authors:
  # - name: Adrian M. Price-Whelan^[Co-first author] # note this makes a footnote saying 'Co-first author'
  #   orcid: 0000-0000-0000-0000
  #   affiliation: "1, 2" # (Multiple affiliations must be quoted)
  # - name: Author Without ORCID^[Co-first author] # note this makes a footnote saying 'Co-first author'
  #   affiliation: 2
  # - name: Author with no affiliation^[Corresponding author]
  #   affiliation: 3
  - name: Peter St. John
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Charles Edison Tripp
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Struan Clark
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: David Biagioni
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: National Renewable Energy Laboratory
   index: 1
#  - name: Institution Name, Country
#    index: 2
#  - name: Independent Researcher, Country
#    index: 3
date: 17 May 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Many important and challenging problems in combinatorial optimization (CO) can 
be expressed as graph search problems in which graph vertices represent full or 
partial solutions and edges represent decisions that connect them.  Graph 
structure not only introduces strong _relational inductive biases_ for learning
[@battaglia2018relational], but lends itself to problems both with and without 
clearly defined algebraic structure.  For example, classic CO problems on graphs such as the traveling salesman problem (TSP) can be expressed as either pure graph search _or_ integer program with well defined linear objective function and linear constraints.  Other problems, however, such as molecular optimization, do no have concise algebraic formulations and yet are readily implemented as graph search [@sv2021multi].  In recent  years, reinforcement learning (RL) has emerged as an effective paradigm for optimizing searches over graphs and led to state-of-the-art heuristics for games like Go and chess, as well as for classical CO problems such as the Traveling Salesman Problem (TSP).  This combination of graph search and RL, while powerful, requires non-trivial
software to execute, especially when combining advanced state representations such as Graph Neural Networks (GNN) with scalable RL algorithms.

# Statement of need

The `graphenv` Python library is designed to 1) make graph search problems more readily expressible as RL problems via an extension of the OpenAI gym API [@brockman2016openai] while 2) enabling their solution via scalable learning algorithms in the popular RLLib library [@liang2018rllib].  

<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@biagioni2020rlmolecule`  ->  "Author et al. (2001)"
- `[@biagioni2020rlmolecule]` -> "(Author et al., 2001)"
- `[@biagioni2020rlmolecule; @sv2021multi]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures -->
<!-- 
Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->


# Acknowledgements

TODO

# References

