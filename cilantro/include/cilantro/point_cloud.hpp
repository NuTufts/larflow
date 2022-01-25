#pragma once

#include <iterator>
#include <set>
#include <cilantro/image_point_cloud_conversions.hpp>
#include <cilantro/grid_downsampler.hpp>
#include <cilantro/normal_estimation.hpp>
#include <cilantro/io.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointCloud {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        VectorSet<ScalarT,EigenDim> points;
        VectorSet<ScalarT,EigenDim> normals;
        VectorSet<float,3> colors;

        PointCloud() {}

        PointCloud(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points) : points(points) {}

        PointCloud(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                   const ConstVectorSetMatrixMap<float,3> &colors)
                : points(points), normals(normals), colors(colors)
        {}

        PointCloud(const PointCloud<ScalarT,EigenDim> &cloud,
                   const std::vector<size_t> &indices,
                   bool negate = false)
        {
            std::set<size_t> indices_set;
            if (negate) {
                std::vector<size_t> full_indices(cloud.size());
                for (size_t i = 0; i < cloud.size(); i++) full_indices[i] = i;
                std::set<size_t> indices_to_discard(indices.begin(), indices.end());
                std::set_difference(full_indices.begin(), full_indices.end(), indices_to_discard.begin(), indices_to_discard.end(), std::inserter(indices_set, indices_set.begin()));
            } else {
                indices_set = std::set<size_t>(indices.begin(), indices.end());
            }

            if (cloud.hasNormals() && cloud.hasColors()) {
                points.resize(cloud.points.rows(), indices_set.size());
                normals.resize(cloud.normals.rows(), indices_set.size());
                colors.resize(3, indices_set.size());
                size_t k = 0;
                for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
                    points.col(k) = cloud.points.col(*it);
                    normals.col(k) = cloud.normals.col(*it);
                    colors.col(k++) = cloud.colors.col(*it);
                }
            } else if (cloud.hasNormals()) {
                points.resize(cloud.points.rows(), indices_set.size());
                normals.resize(cloud.normals.rows(), indices_set.size());
                size_t k = 0;
                for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
                    points.col(k) = cloud.points.col(*it);
                    normals.col(k++) = cloud.normals.col(*it);
                }
            } else if (cloud.hasColors()) {
                points.resize(cloud.points.rows(), indices_set.size());
                colors.resize(3, indices_set.size());
                size_t k = 0;
                for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
                    points.col(k) = cloud.points.col(*it);
                    colors.col(k++) = cloud.colors.col(*it);
                }
            } else {
                points.resize(cloud.points.rows(), indices_set.size());
                size_t k = 0;
                for (auto it = indices_set.begin(); it != indices_set.end(); ++it) {
                    points.col(k++) = cloud.points.col(*it);
                }
            }
        }

        template <class DepthConverterT, class = typename std::enable_if<EigenDim == 3 && std::is_same<typename DepthConverterT::MetricDepth,ScalarT>::value>::type>
        PointCloud(const typename DepthConverterT::RawDepth* depth_data,
                   const DepthConverterT &depth_converter,
                   size_t image_w, size_t image_h,
                   const Eigen::Ref<const Eigen::Matrix<ScalarT,3,3>> &intrinsics,
                   bool keep_invalid = false)
        {
            depthImageToPoints<DepthConverterT>(depth_data, depth_converter, image_w, image_h, intrinsics, points, keep_invalid);
        }

        template <class DepthConverterT, class = typename std::enable_if<EigenDim == 3 && std::is_same<typename DepthConverterT::MetricDepth,ScalarT>::value>::type>
        PointCloud(unsigned char* rgb_data,
                   const typename DepthConverterT::RawDepth* depth_data,
                   const DepthConverterT &depth_converter,
                   size_t image_w, size_t image_h,
                   const Eigen::Ref<const Eigen::Matrix<ScalarT,3,3>> &intrinsics,
                   bool keep_invalid = false)
        {
            RGBDImagesToPointsColors<DepthConverterT>(rgb_data, depth_data, depth_converter, image_w, image_h, intrinsics, points, colors, keep_invalid);
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == 3>::type>
        PointCloud(const std::string &file_name) {
            readPointCloudFromPLYFile<ScalarT>(file_name, points, normals, colors);
        }

        inline size_t size() const { return points.cols(); }

        inline bool hasNormals() const { return points.cols() > 0 && normals.cols() == points.cols(); }

        inline bool hasColors() const { return points.cols() > 0 && colors.cols() == points.cols(); }

        inline bool isEmpty() const { return points.cols() == 0; }

        PointCloud& clear() {
            points.resize(Eigen::NoChange, 0);
            normals.resize(Eigen::NoChange, 0);
            colors.resize(Eigen::NoChange, 0);
            return *this;
        }

        PointCloud& append(const PointCloud<ScalarT,EigenDim> &cloud) {
            size_t original_size = size();

            points.conservativeResize(cloud.points.rows(), original_size + cloud.points.cols());
            points.rightCols(cloud.points.cols()) = cloud.points;
            if (normals.cols() == original_size && cloud.hasNormals()) {
                normals.conservativeResize(cloud.normals.rows(), original_size + cloud.normals.cols());
                normals.rightCols(cloud.normals.cols()) = cloud.normals;
            }
            if (colors.cols() == original_size && cloud.hasColors()) {
                colors.conservativeResize(3, original_size + cloud.colors.cols());
                colors.rightCols(cloud.colors.cols()) = cloud.colors;
            }
            return *this;
        }

        PointCloud& remove(const std::vector<size_t> &indices) {
            if (indices.empty()) return *this;

            std::set<size_t> indices_set(indices.begin(), indices.end());
            if (indices_set.size() >= size()) {
                clear();
                return *this;
            }

            size_t valid_ind = size() - 1;
            while (indices_set.find(valid_ind) != indices_set.end()) {
                valid_ind--;
            }

            const bool has_normals = hasNormals();
            const bool has_colors = hasColors();

            auto ind_it = indices_set.begin();
            while (ind_it != indices_set.end() && *ind_it < valid_ind) {
                points.col(*ind_it).swap(points.col(valid_ind));
                if (has_normals) {
                    normals.col(*ind_it).swap(normals.col(valid_ind));
                }
                if (has_colors) {
                    colors.col(*ind_it).swap(colors.col(valid_ind));
                }
                valid_ind--;
                while (*ind_it < valid_ind && indices_set.find(valid_ind) != indices_set.end()) {
                    valid_ind--;
                }
                ++ind_it;
            }

            const size_t original_size = size();
            const size_t new_size = valid_ind + 1;
            points.conservativeResize(Eigen::NoChange, new_size);
            if (normals.cols() == original_size) {
                normals.conservativeResize(Eigen::NoChange, new_size);
            }
            if (colors.cols() == original_size) {
                colors.conservativeResize(Eigen::NoChange, new_size);
            }

            return *this;
        }

        PointCloud& removeInvalidPoints() {
            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                if (!points.col(i).allFinite()) ind_to_remove.emplace_back(i);
            }
            return remove(ind_to_remove);
        }

        PointCloud& removeInvalidNormals() {
            if (!hasNormals()) return *this;

            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(normals.cols());
            for (size_t i = 0; i < normals.cols(); i++) {
                if (!normals.col(i).allFinite()) ind_to_remove.emplace_back(i);
            }
            return remove(ind_to_remove);
        }

        PointCloud& removeInvalidColors() {
            if (!hasColors()) return *this;

            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(colors.cols());
            for (size_t i = 0; i < colors.cols(); i++) {
                if (!colors.col(i).allFinite()) ind_to_remove.emplace_back(i);
            }
            return remove(ind_to_remove);
        }

        PointCloud& removeInvalidData() {
            const bool has_normals = hasNormals();
            const bool has_colors = hasColors();

            std::vector<size_t> ind_to_remove;
            ind_to_remove.reserve(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                if (!points.col(i).allFinite() || (has_normals && !normals.col(i).allFinite()) || (has_colors && !colors.col(i).allFinite())) {
                    ind_to_remove.emplace_back(i);
                }
            }
            return remove(ind_to_remove);
        }

        PointCloud& gridDownsample(ScalarT bin_size, size_t min_points_in_bin = 1) {
            if (hasNormals() && hasColors()) {
                PointsNormalsColorsGridDownsampler<ScalarT,EigenDim>(points, normals, colors, bin_size).getDownsampledPointsNormalsColors(points, normals, colors, min_points_in_bin);
            } else if (hasNormals()) {
                PointsNormalsGridDownsampler<ScalarT,EigenDim>(points, normals, bin_size).getDownsampledPointsNormals(points, normals, min_points_in_bin);
            } else if (hasColors()) {
                PointsColorsGridDownsampler<ScalarT,EigenDim>(points, colors, bin_size).getDownsampledPointsColors(points, colors, min_points_in_bin);
            } else {
                PointsGridDownsampler<ScalarT,EigenDim>(points, bin_size).getDownsampledPoints(points, min_points_in_bin);
            }
            return *this;
        }

        PointCloud gridDownsampled(ScalarT bin_size, size_t min_points_in_bin = 1) const {
            PointCloud res;
            if (hasNormals() && hasColors()) {
                PointsNormalsColorsGridDownsampler<ScalarT,EigenDim>(points, normals, colors, bin_size).getDownsampledPointsNormalsColors(res.points, res.normals, res.colors, min_points_in_bin);
            } else if (hasNormals()) {
                PointsNormalsGridDownsampler<ScalarT,EigenDim>(points, normals, bin_size).getDownsampledPointsNormals(res.points, res.normals, min_points_in_bin);
            } else if (hasColors()) {
                PointsColorsGridDownsampler<ScalarT,EigenDim>(points, colors, bin_size).getDownsampledPointsColors(res.points, res.colors, min_points_in_bin);
            } else {
                PointsGridDownsampler<ScalarT,EigenDim>(points, bin_size).getDownsampledPoints(res.points, min_points_in_bin);
            }
            return res;
        }

        inline PointCloud& estimateNormalsKNN(size_t k) {
            normals = NormalEstimation<ScalarT,EigenDim>(points).estimateNormalsKNN(k);
            return *this;
        }

        inline PointCloud& estimateNormalsKNN(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree,
                                              size_t k)
        {
            normals = NormalEstimation<ScalarT,EigenDim>(kd_tree).estimateNormalsKNN(k);
            return *this;
        }

        inline PointCloud& estimateNormalsRadius(ScalarT radius) {
            normals = NormalEstimation<ScalarT,EigenDim>(points).estimateNormalsRadius(radius);
            return *this;
        }

        inline PointCloud& estimateNormalsRadius(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree,
                                                 ScalarT radius)
        {
            normals = NormalEstimation<ScalarT,EigenDim>(kd_tree).estimateNormalsRadius(radius);
            return *this;
        }

        inline PointCloud& estimateNormalsKNNInRadius(size_t k, ScalarT radius) {
            normals = NormalEstimation<ScalarT,EigenDim>(points).estimateNormalsKNNInRadius(k, radius);
            return *this;
        }

        inline PointCloud& estimateNormalsKNNInRadius(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree,
                                                      size_t k,
                                                      ScalarT radius)
        {
            normals = NormalEstimation<ScalarT,EigenDim>(kd_tree).estimateNormalsKNNInRadius(k, radius);
            return *this;
        }

        inline PointCloud& estimateNormals(const NeighborhoodSpecification<ScalarT> &nh) {
            normals = NormalEstimation<ScalarT,EigenDim>(points).estimateNormals(nh);
            return *this;
        }

        inline PointCloud& estimateNormals(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree,
                                           const NeighborhoodSpecification<ScalarT> &nh)
        {
            normals = NormalEstimation<ScalarT,EigenDim>(kd_tree).estimateNormals(nh);
            return *this;
        }

        inline PointCloud& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &rotation,
                                     const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1>> &translation)
        {
            points = (rotation*points).colwise() + translation;
            if (hasNormals()) normals = rotation*normals;
            return *this;
        }

        inline PointCloud transformed(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &rotation,
                                      const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1>> &translation)
        {
            PointCloud cloud;
            cloud.points.noalias() = (rotation*points).colwise() + translation;
            if (hasNormals()) cloud.normals.noalias() = rotation*normals;
            if (hasColors()) cloud.colors = colors;
            return cloud;
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        inline PointCloud& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim+1,EigenDim+1>> &tform) {
            return transform(tform.topLeftCorner(EigenDim,EigenDim), tform.topRightCorner(EigenDim,1));
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        inline PointCloud transformed(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim+1,EigenDim+1>> &tform) {
            return transformed(tform.topLeftCorner(EigenDim,EigenDim), tform.topRightCorner(EigenDim,1));
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        inline PointCloud& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &tform) {
            return transform(tform.topLeftCorner(points.rows(),points.rows()), tform.topRightCorner(points.rows(),1));
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        inline PointCloud transformed(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &tform) {
            return transformed(tform.topLeftCorner(points.rows(),points.rows()), tform.topRightCorner(points.rows(),1));
        }

        inline PointCloud& transform(const RigidTransformation<ScalarT,EigenDim> &tform) {
            points = tform*points;
            if (hasNormals()) normals = tform.linear()*normals;
            return *this;
        }

        inline PointCloud transformed(const RigidTransformation<ScalarT,EigenDim> &tform) const {
            PointCloud cloud;
            cloud.points.noalias() = tform*points;
            if (hasNormals()) cloud.normals.noalias() = tform.linear()*normals;
            if (hasColors()) cloud.colors = colors;
            return cloud;
        }

        inline PointCloud& transform(const RigidTransformationSet<ScalarT,EigenDim> &tforms) {
            if (hasNormals()) {
                tforms.warpPointsNormals(points, normals);
            } else {
                tforms.warpPoints(points);
            }
            return *this;
        }

        inline PointCloud transformed(const RigidTransformationSet<ScalarT,EigenDim> &tforms) const {
            PointCloud cloud(*this);
            cloud.transform(tforms);
            return cloud;
        }

        template <class DepthConverterT, class = typename std::enable_if<EigenDim == 3 && std::is_same<typename DepthConverterT::MetricDepth,ScalarT>::value>::type>
        inline PointCloud& fromDepthImage(const typename DepthConverterT::RawDepth* depth_data,
                                          const DepthConverterT &depth_converter,
                                          size_t image_w, size_t image_h,
                                          const Eigen::Ref<const Eigen::Matrix<ScalarT,3,3>> &intrinsics,
                                          bool keep_invalid = false)
        {
            normals.resize(Eigen::NoChange, 0);
            colors.resize(Eigen::NoChange, 0);
            depthImageToPoints<DepthConverterT>(depth_data, depth_converter, image_w, image_h, intrinsics, points, keep_invalid);
            return *this;
        }

        template <class DepthConverterT, class = typename std::enable_if<EigenDim == 3 && std::is_same<typename DepthConverterT::MetricDepth,ScalarT>::value>::type>
        inline PointCloud& fromRGBDImages(unsigned char* rgb_data,
                                          const typename DepthConverterT::RawDepth* depth_data,
                                          const DepthConverterT &depth_converter,
                                          size_t image_w, size_t image_h,
                                          const Eigen::Ref<const Eigen::Matrix<ScalarT,3,3>> &intrinsics,
                                          bool keep_invalid = false)
        {
            normals.resize(Eigen::NoChange, 0);
            RGBDImagesToPointsColors<DepthConverterT>(rgb_data, depth_data, depth_converter, image_w, image_h, intrinsics, points, colors, keep_invalid);
            return *this;
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == 3>::type>
        inline PointCloud& fromPLYFile(const std::string &file_name) {
            readPointCloudFromPLYFile<ScalarT>(file_name, points, normals, colors);
            return *this;
        }

        template <typename ScalarOutT = ScalarT, ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == 3>::type>
        inline const PointCloud& toPLYFile(const std::string &file_name, bool binary = true) const {
            writePointCloudToPLYFile<ScalarT,ScalarOutT>(file_name, points, normals, colors, binary);
            return *this;
        }

        template <typename NewScalarT>
        PointCloud<NewScalarT,EigenDim> cast() const {
            PointCloud<NewScalarT,EigenDim> res;
            res.points = points.template cast<NewScalarT>();
            res.normals = normals.template cast<NewScalarT>();
            res.colors = colors;
            return res;
        }
    };

    typedef PointCloud<float,2> PointCloud2f;
    typedef PointCloud<double,2> PointCloud2d;
    typedef PointCloud<float,3> PointCloud3f;
    typedef PointCloud<double,3> PointCloud3d;
    typedef PointCloud<float,Eigen::Dynamic> PointCloudXf;
    typedef PointCloud<double,Eigen::Dynamic> PointCloudXd;
}
