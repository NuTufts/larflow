
.. _program_listing_file_larflow_KeyPoints_README.md:

Program Listing for File README.md
==================================

|exhale_lsh| :ref:`Return to documentation for file <file_larflow_KeyPoints_README.md>` (``larflow/KeyPoints/README.md``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: markdown

   # PrepKeyPointData
   
   Goal of this module is to provide tools to build up Keypoint datasets.
   
   The inspiration is the [OpenPose network](https://arxiv.org/pdf/1812.08008.pdf). This predicts two quantities:
   
   * a confidence map for the location of keypoints and
   * a flow field that predictions the direction from one keypoint to another.
   
   This module prepares the ground truth for the application of the OpenPose strategy to LArTPC particle reconstruction.
   We define keypoints as
   
   * track start and end points
   * shower start points
   * neutrino interactions
   
   We define a flow field, only one for all particles.
   This field consists of the direction the particle is traveling at a given pixel or spacepoint.
   
   ## Example of keypoint ground truth
   
   This will come in the form of labels for proposed 3D spacepoints.
   
   ![Demo](https://raw.githubusercontent.com/NuTufts/larflow/ubdl_dev/larflow/KeyPoints/test/demo_prepkeypoint.png)
   
   ## Example of Flow Field ground truth
   
   ![FlowDemo](https://raw.githubusercontent.com/NuTufts/larflow/ubdl_dev/larflow/KeyPoints/test/demo_flowfield3d.png)
   
   ![FlowDemoZoomIn](https://raw.githubusercontent.com/NuTufts/larflow/ubdl_dev/larflow/KeyPoints/test/demo_flowfield3dzoomin.png)
   
   
