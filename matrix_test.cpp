//#define EIGEN_USE_MKL_ALL

#include <iostream>
//#include<vector>
#include <fstream>
#include <Eigen\Sparse>
//#include<Eigen/IterativeLinearSolvers>
//#include <Eigen/SparseCholesky>
//#include <Eigen/SparseLU>
#include <ctime>

#define xNodeNumber 81
#define yNodeNumber 81
#define dx 0.01
#define dy 0.01
#define eps0 8.85e-12
#define rho 0
#define potential 100

using namespace std;
using namespace Eigen;
//typedef Eigen::Triplet<double> T;
//typedef Eigen::SparseMatrix<double> SparseMatrixType;


int main()
{
		int nCores = 12;
		omp_set_num_threads(nCores);
		setNbThreads(nCores);
		double t0 = clock();
		int allNodeNumber = xNodeNumber * yNodeNumber;
		int allElementNumber = (xNodeNumber - 1) * (yNodeNumber - 1) * 2;
		int i, j;
		int nNode = 0;
		int nElement = 0;
		double Area = 0.5 * dx * dy;

		//定义数组
		double Ke[3][3], be[3];
		double bn[3], cn[3];
		double* xAllNode = new double[allNodeNumber];
		double* yAllNode = new double[allNodeNumber];
		int** allNodeIdentifier = new int* [allNodeNumber];
		for (i = 0; i < xNodeNumber; i++)
		{
				allNodeIdentifier[i] = new int[yNodeNumber];
		}
		int* allElementIdentifier = new int[allElementNumber];
		int** elementNodeIdentifier = new int* [3];
		for (i = 0; i < 3; i++)
		{
				elementNodeIdentifier[i] = new int[allElementNumber];
		}
		SparseMatrix<double, RowMajor> K(allNodeNumber, allNodeNumber);
		SparseMatrix<double, RowMajor> Z(1, allNodeNumber);
		//SparseMatrix<double, RowMajor> U(1, allNodeNumber);
		VectorXd b, U;
		b = VectorXd::Zero(allNodeNumber);
		U = VectorXd::Zero(allNodeNumber);


		//所有节点编号
		for (j = 0; j < yNodeNumber; j++)
		{
				for (i = 0; i < xNodeNumber; i++)
				{
						allNodeIdentifier[i][j] = nNode;
						xAllNode[nNode] = i * dx;
						yAllNode[nNode] = j * dy;
						nNode++;
				}
		}

		//单元节点编号对应所有节点编号
		for (j = 0; j < yNodeNumber - 1; j++)
		{
				for (i = 0; i < xNodeNumber - 1; i++)
				{
						elementNodeIdentifier[0][nElement] = allNodeIdentifier[i][j + 1];
						elementNodeIdentifier[1][nElement] = allNodeIdentifier[i][j];
						elementNodeIdentifier[2][nElement] = allNodeIdentifier[i + 1][j];
						nElement++;
						elementNodeIdentifier[0][nElement] = allNodeIdentifier[i][j + 1];
						elementNodeIdentifier[1][nElement] = allNodeIdentifier[i + 1][j];
						elementNodeIdentifier[2][nElement] = allNodeIdentifier[i + 1][j + 1];
						nElement++;
				}
		}

		//计算Ke,be,并写入K,b
//#pragma omp parallel for
		for (nElement = 0; nElement < allElementNumber; nElement++)
		{
				int n0, n1, n2;
				//double a, b, c;
				n0 = elementNodeIdentifier[0][nElement];
				n1 = elementNodeIdentifier[1][nElement];
				n2 = elementNodeIdentifier[2][nElement];
				bn[0] = yAllNode[n1] - yAllNode[n2];
				bn[1] = yAllNode[n2] - yAllNode[n0];
				bn[2] = yAllNode[n0] - yAllNode[n1];
				cn[0] = xAllNode[n2] - xAllNode[n1];
				cn[1] = xAllNode[n0] - xAllNode[n2];
				cn[2] = xAllNode[n1] - xAllNode[n0];
				for (i = 0; i < 3; i++)
				{
						int iTemp = elementNodeIdentifier[i][nElement];
						for (j = 0; j < 3; j++)
						{
								Ke[i][j] = (bn[i] * bn[j] + cn[i] * cn[j]) / (4 * Area);
								double t = (bn[i] * bn[j] + cn[i] * cn[j]) / (4 * Area);
								int jTemp = elementNodeIdentifier[j][nElement];
								K.coeffRef(iTemp, jTemp) += Ke[i][j];
						}
						be[i] = Area / 3 * rho / eps0;
						b[iTemp] += be[i];
				}
		}
		//cout << "mat.nonZeros() = " << K.nonZeros() << endl;
		double t1 = clock();
		double delta_t1 = (t1 - t0) / CLOCKS_PER_SEC;
		cout << "t1 = " << delta_t1 << endl;

		//设置边界条件
		int temp;
		for (i = 0; i < xNodeNumber; i++)
		{
				temp = allNodeIdentifier[i][0];
				Z.coeffRef(0, temp) = 1;
				K.row(temp) = Z;
				Z.setZero();
				b[temp] = 0;

				temp = allNodeIdentifier[i][yNodeNumber - 1];
				Z.coeffRef(0, temp) = 1;
				K.row(temp) = Z;
				Z.setZero();
				b[temp] = 0;
		}
		for (j = 1; j < xNodeNumber - 1; j++)
		{
				temp = allNodeIdentifier[xNodeNumber - 1][j];
				Z.coeffRef(0, temp) = 1;
				K.row(temp) = Z;
				Z.setZero();
				b[temp] = 0;
		}
		for (i = 0; i < int(xNodeNumber / 4); i++)
		{
				temp = allNodeIdentifier[i][int(yNodeNumber / 2)];
				Z.coeffRef(0, temp) = 1;
				K.row(temp) = Z;
				Z.setZero();
				b[temp] = potential;
		}
		//cout << "mat.nonZeros() = " << K.nonZeros() << endl;
		double t2 = clock();
		double delta_t2 = (t2 - t1) / CLOCKS_PER_SEC;
		cout << "t2 = " << delta_t2 << endl;

		//SparseLU<SparseMatrix<double>> solver;
		BiCGSTAB<SparseMatrix<double, RowMajor>> solver;
		//LeastSquaresConjugateGradient<SparseMatrix<double>> solver;
		solver.compute(K);
//#pragma omp parallel
		U = solver.solve(b);
		//cout << "vec.nonZeros() = " << U.nonZeros() << endl;
		double t3 = clock();
		double delta_t3 = (t3 - t2) / CLOCKS_PER_SEC;
		cout << "t3 = " << delta_t3 << endl;

		double delta_t = (t3 - t0) / CLOCKS_PER_SEC;
		cout << "t_total = " << delta_t << endl;



		//输出结果
		ofstream fp("Potential.dat", ofstream::out);
		fp << "title=\"Potential\"" << endl;
		fp << "variables = \"x\",\"y\",\"U(V)\"" << endl;
		fp << "zone i=" << xNodeNumber << ",j=" << yNodeNumber << endl;
		for (j = 0; j < yNodeNumber; j++)
		{
				for (i = 0; i < xNodeNumber; i++)
				{
						fp << i * dx << " " << j * dx << " " << U(i + j * xNodeNumber) << endl;
				}
		}
		fp.close();


		//SparseMatrix<double, RowMajor> mat(1000, 2000);//创建一个行优先的，维度为1000x2000的稀疏矩阵
		//SparseVector<double, RowMajor> vec(1000);
		////标准维度
		//cout << "mat.rows() = " << mat.rows() << endl;
		//cout << "mat.cols() = " << mat.cols() << endl;
		//cout << "mat.size() = " << mat.size() << endl;
		//cout << "vec.size() = " << vec.size() << endl;
		////内/外维度
		//cout << "mat.innerSize() = " << mat.innerSize() << endl; //行优先，所以为列数
		//cout << "mat.outerSize() = " << mat.outerSize() << endl;
		////非零元素个数
		//cout << "mat.nonZeros() = " << mat.nonZeros() << endl;
		//cout << "vec.nonZeros() = " << vec.nonZeros() << endl;
		//return 0;


		//SparseMatrix<double, RowMajor> mat(5, 7);//创建一个行优先的，维度为1000x2000的稀疏矩阵
		////随机访问（读/写）元素
		//cout << mat.coeffRef(0, 0) << endl; //读取元素
		//mat.coeffRef(0, 0) = 5;
		//cout << mat.coeffRef(0, 0) << endl; //写入元素

		////迭代访问稀疏矩阵
		//cout << "\n迭代访问稀疏矩阵的元素：" << endl;
		//for (int k = 0; k < mat.outerSize(); ++k)
		//		for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
		//		{
		//				it.value(); // 元素值
		//				it.row();   // 行标row index
		//				it.col();   // 列标（此处等于k）
		//				it.index(); // 内部索引，此处等于it.row()
		//		}
		//cout << mat << endl;

		////迭代访问稀疏向量
		//SparseVector<double, RowMajor> vec(5);
		////vec.coeffRef(0,0) = 1;
		//for (SparseVector<double>::InnerIterator it(vec); it; ++it)
		//{
		//		it.value(); // == vec[ it.index() ]
		//		it.index(); //索引
		//}
		//cout << vec << endl;
		//return 0;


		//int rows = 10, cols = 10;
		//int estimation_of_entries = 10; //预计非零元素的个数
		//std::vector<T> tripletList;
		//tripletList.reserve(estimation_of_entries);
		//int j = 1; // 列标
		//for (int i = 0; i < estimation_of_entries; i++) { //遍历行

		//		int v_ij = i * i + 1;
		//		tripletList.push_back(T(i, j, v_ij));
		//}
		//SparseMatrixType mat(rows, cols);
		//mat.setFromTriplets(tripletList.begin(), tripletList.end()); //根据三元组列表生成稀疏矩阵
		//// 
		//cout << mat << endl;
		//return 0;


		//int rows = 10, cols = 10;
		//SparseMatrix<double> mat(rows, cols);         // 默认列优先
		//mat.reserve(VectorXi::Constant(cols, 6)); //关键：为每一列保留6个非零元素空间
		//for (int i = 0; i < 3; i++) { //遍历行
		//		for (int j = 0; j < 3; j++) {
		//				int v_ij = i + j + 1;
		//				mat.insert(i, j) = v_ij;                    // alternative: mat.coeffRef(i,j) += v_ij;
		//		}
		//}
		//mat.makeCompressed(); //压缩剩余的空间
		//cout << mat << endl;
		//return 0;
}