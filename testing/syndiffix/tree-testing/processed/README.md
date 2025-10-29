## 1 dim
These tests show:

1. That the use of range extending and leaf adding is not significantly different from traditional 1dim approach.
2. That range extending in particular seems to have very little effect, and can probably be ignored (for 1dim).
3. That both the simple and leafs-only approaches to leaf adding are significantly better than no leaf adding, and that simple is often slightly better than leafs-only. However, leafs-only is the safer alternative, and the tail-extending effect of simple can in any event be artificially generated later in the pipeline (at microdata time, after the leafs are built).