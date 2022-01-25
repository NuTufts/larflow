#include <cilantro/normal_estimation.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>
#include <cilantro/timer.hpp>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f cloud(argv[1]);

    if (cloud.isEmpty()) {
        std::cout << "Input cloud is empty!" << std::endl;
        return 0;
    }

    // Clear input normals
    cloud.normals.resize(Eigen::NoChange, 0);

    cloud.gridDownsample(0.005f);

    cilantro::Timer tree_timer;
    tree_timer.start();
    cilantro::KDTree3f tree(cloud.points);
//    cilantro::NormalEstimation3f ne(tree);
//    cilantro::NormalEstimation3f ne(cloud.points);
    tree_timer.stop();

    cilantro::Timer ne_timer;
    ne_timer.start();

//    cloud.normals = ne.estimateNormals(cilantro::NeighborhoodSpecification<float>(cilantro::NeighborhoodType::KNN_IN_RADIUS, 7, 0.01f));
//    cloud.normals = ne.estimateNormalsKNNInRadius(7, 0.01f);
//    cloud.normals = ne.estimateNormalsRadius(0.01f);
//    cloud.normals = ne.estimateNormalsKNN(7);

//    cloud.estimateNormals(tree, cilantro::NeighborhoodSpecification<float>(cilantro::NeighborhoodType::KNN_IN_RADIUS, 7, 0.01f));
//    cloud.estimateNormalsKNNInRadius(tree, 7, 0.01f);
//    cloud.estimateNormalsRadius(tree, 0.01f);
    cloud.estimateNormalsKNN(tree, 7);

    // Search tree argument is optional (automatically built):
//    cloud.estimateNormals(cilantro::NeighborhoodSpecification<float>(cilantro::NeighborhoodType::KNN_IN_RADIUS, 7, 0.01f));
//    cloud.estimateNormalsKNNInRadius(7, 0.01f);
//    cloud.estimateNormalsRadius(0.01f);
//    cloud.estimateNormalsKNN(7);

    ne_timer.stop();

    std::cout << "kd-tree time: " << tree_timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Estimation time: " << ne_timer.getElapsedTime() << "ms" << std::endl;

    cilantro::Visualizer viz("NormalEstimation example", "disp");

    viz.addObject<cilantro::PointCloudRenderable>("cloud_d", cloud, cilantro::RenderingProperties().setDrawNormals(true));

    std::cout << "Press 'n' to toggle rendering of normals" << std::endl;
    while (!viz.wasStopped()){
        viz.spinOnce();
    }

    return 0;
}
