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
  - name: David Biagioni
    orcid: 0000-0001-6140-1957
    affiliation: 2
  - name: Charles Edison Tripp
    orcid: 0000-0002-5867-3561
    affiliation: 2
  - name: Struan Clark
    affiliation: 2
    orcid:  0000-0003-0078-6560
  - name: Dmitry Duplyakin
    affiliation: 2
    orcid:  0000-0001-5132-0168
  - name: Jeffrey Law
    affiliation: 1
    orcid: 0000-0003-2828-1273    
  - name: Peter St. John #^[Corresponding author]
    orcid: 0000-0002-7928-3722
    corresponding: true # (This is how to denote the corresponding author)    
    affiliation: 1    
affiliations:
 - name: Biosciences Center, National Renewable Energy Laboratory, Golden CO 80401, USA
   index: 1
 - name: Computational Sciences Center, National Renewable Energy Laboratory, Golden CO 80401, USA
   index: 2

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

Many important and challenging problems in combinatorial optimization (CO) can be
expressed as graph search problems, in which graph vertices represent full or partial
solutions and edges represent decisions that connect them. 
Graph structure not only introduces strong _relational inductive biases_ for learning [@battaglia2018relational], but lends itself to problems both with and without clearly defined algebraic structure.
For example, classic CO problems on graphs such as the traveling salesman problem (TSP)
can be expressed as either pure graph search _or_ integer program with well defined
linear objective function and linear constraints.  Other problems, however, such as
molecular optimization, do no have concise algebraic formulations and yet are readily
implemented as a graph search [@Zhou_2019;@sv2021multi].  In recent  years, reinforcement learning
(RL) has emerged as an effective paradigm for optimizing searches over graphs and led to
state-of-the-art heuristics for games like Go and chess, as well as for classical CO
problems such as the Traveling Salesman Problem (TSP).  This combination of graph search
and RL, while powerful, requires non-trivial software to execute, especially when
combining advanced state representations such as Graph Neural Networks (GNN) with
scalable RL algorithms.

# Statement of need

The `graphenv` Python library is designed to 1) make graph search problems more readily
expressible as RL problems via an extension of the OpenAI gym API [@brockman2016openai]
while 2) enabling their solution via scalable learning algorithms in the popular RLLib
library [@liang2018rllib].  

RLLib provides out-of-the-box support for both parametrically-defined actions and
masking of invalid actions. However, native support for action spaces where the action
_choices_ change for each state is challenging to implement in a computationally
efficient fashion. The `graphenv` library provides utility classes that simplify the
flattening and masking of action observations for choosing from a set of successor
states at every node in a graph search.

Related software efforts have addressed parts of the above need. OpenGraphGym [@Zheng_2020] implements RL-based stragies for common graph optimization challenges such as minimum vertex cover or maximum cut, but does not interface with external RL libraries and has minimal documentation. Ecole [@prouvost2020ecole] provides an OpenAI-like gym environment for combinatorial optimization, but intends to operate in concert with traditional mixed integer solvers rather than directly exposing the environment to an RL agent.


# Examples of usage

This package is a generalization of methods employed in the optimization of molecular
structure for energy storage applications, funded by US Department of Energy (DOE)'s Advanced Research Projects
Agency - Energy [@sv2021multi]. Specifically, this package enables optimization against
a surrogate objective function based on high-throughput density functional theory
calculations [@Sowndarya_S_V_2021; @St_John_2020_a; @St_John_2020_b] by considering
molecule selection as an iterative process of adding atoms and bonds, transforming the
optimization into a rooted search over a directed, acyclic graph.  Ongoing work is
leveraging this library to enable similar optimization for inorganic crystal structures,
again using a surrogate objective function based on high-throughput quantum mechanical
calculations [@Pandey_2021].


# Acknowledgements

This work was authored by the National Renewable Energy Laboratory, operated by Alliance
for Sustainable Energy, LLC, for the US Department of Energy (DOE) under Contract No.
DE-AC36-08GO28308. The information, data, or work presented herein was funded in part by
the Advanced Research Projects Agency-Energy (ARPA-E), U.S. Department of Energy, under
Award Number DE-AR0001205. The views and opinions of authors expressed herein do not
necessarily state or reflect those of the United States Government or any agency
thereof. The US Government retains and the publisher, by accepting the article for
publication, acknowledges that the US Government retains a nonexclusive, paid-up,
irrevocable, worldwide license to publish or reproduce the published form of this work
or allow others to do so, for US Government purposes.

# References



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
