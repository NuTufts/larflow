
.. _program_listing_file_larflow_Reco_README.md:

Program Listing for File README.md
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_larflow_Reco_README.md>` (``larflow/Reco/README.md``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: markdown

   # LArFlow Reconstruction tools
   
   This code is for track, shower, and vertex reconstruction using larmatch points as inputs.
   
   ## strategy
   
   We perform a merge/break loop with the clusters
   
   Prep stage
   
   * first make clusters using dbscan
   * select clusters of minimum size
   
   Break/Merge loop
   
   * next get pca for all clusters
   * break individual clusters at point greatest distance to first pca-axis
   * re-pca broken clusters
   * merge clusters if on same line: pca-axis intersects at point between. point between some min distance from line between cluster ends.
   * reform cluster and re-pca
   * stop when no more clusters merge or break
