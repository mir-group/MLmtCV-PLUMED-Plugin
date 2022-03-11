
#include <cmath>

#include "tf_utils.hpp"
#include <tensorflow/c/c_api.h> // TensorFlow C API header

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
 
using namespace std;


int main(int narg, char** argv){

      int n_feature;
      int n_latent;
      static int max_n_latent;

      TF_Graph* graph;
      TF_Status* status;
      TF_Buffer* meta_graph_def;
      TF_Session* sess;
      TF_Output input_op, output_op, grad_op ;

      graph = TF_NewGraph();
      status = TF_NewStatus();
 
      TF_SessionOptions* session_options = TF_NewSessionOptions();
      TF_Buffer* run_options = NULL;
 
      int tags_len = 1;
      char** _tags = (char **) malloc(tags_len * sizeof(char*));
      _tags[0] = (char *) malloc(128 * sizeof(char));
      strcpy(_tags[0], "serve");
      const char* const * tags = _tags;
 
      meta_graph_def = TF_NewBuffer();

      // Load saved model
      char temp[100];
      sprintf(temp, "ANN_TF: Loading export dir %s\n", argv[1]);
      cout<<temp<<endl;
      sess = TF_LoadSessionFromSavedModel(session_options,
                                          run_options,
                                          argv[1],
                                          tags,
                                          tags_len,
                                          graph,
                                          meta_graph_def,
                                          status);

      sprintf(temp, "%s", TF_Message(status));
      cout<<temp<<endl;

      if (graph == nullptr) {
        cout<<"cannot load graph"<<endl;
        return 1;
      } else {
        sprintf(temp, "\n\nANN_TF: Tensorflow graph loaded\n\n");
        cout<<temp<<endl;
      }

      // // print operations in the graph
      // TF_Operation* op;
      // std::size_t pos = 0;
      // while ((op = TF_GraphNextOperation(graph, &pos)) != nullptr) {
      //   const char* name = TF_OperationName(op);
      //   const int num_outputs = TF_OperationNumOutputs(op);
      //   const int num_inputs = TF_OperationNumInputs(op);
      //   sprintf(temp, "%d: name %s input_size %d outputsize %d\n",  pos, name, num_inputs, num_outputs);
      //   cout<<temp<<endl;
      // }

      string i="x", o="nn_return", g="grad_return";
      const char * input = i.c_str();
      const char * outputs = o.c_str();
      const char * grad = g.c_str();

      input_op = {TF_GraphOperationByName(graph, "x"), 0};
      output_op = {TF_GraphOperationByName(graph, outputs), 0};
      grad_op = {TF_GraphOperationByName(graph, grad), 0};
      if (input_op.oper == nullptr) {
        char temp[100];
        sprintf(temp, "Can't find input %s in the graph", input);
        cout<<temp<<endl;
        return 2;
      }
      if (output_op.oper == nullptr) {
        char temp[100];
        sprintf(temp, "Can't find input %s in the graph", outputs);
        cout<<temp<<endl;
        return 2;
      }
      if (grad_op.oper == nullptr) {
        char temp[100];
        sprintf(temp, "Can't find input %s in the graph", grad);
        cout<<temp<<endl;
        return 2;
      }

      // read in input dimensions
      int num_dims = TF_GraphGetTensorNumDims(graph, input_op, status);
      vector<int64_t> dims(num_dims);
      TF_GraphGetTensorShape(graph, input_op, dims.data(), num_dims, status);
      n_feature = dims[1];

      num_dims = TF_GraphGetTensorNumDims(graph, output_op, status);
      TF_GraphGetTensorShape(graph, output_op, dims.data(), num_dims, status);
      n_latent = dims[1];

      cout<<"nfeature "<<n_feature <<" nlatent "<< n_latent<<endl;

      vector<float> input_vals(n_feature);
      input_vals[0] = -1.393487;
      input_vals[1] = 1.796201;
      const vector<int64_t> input_dims = {1, n_feature};
      TF_Tensor* input_tensor = tf_utils::CreateTensor(
              TF_FLOAT, input_dims.data(), input_dims.size(), 
              input_vals.data(), input_vals.size() * sizeof(float));


      TF_Tensor* output_tensor = nullptr;
      TF_SessionRun(sess,
                    nullptr, // Run options.
                    &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                    &output_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                    nullptr, 0, // Target operations, number of targets.
                    nullptr, // Run metadata.
                    status // Output status.
                    );
      const auto input_ref = static_cast<double*>(TF_TensorData(input_tensor));
      const auto output = static_cast<float*>(TF_TensorData(output_tensor));


      // backward  with Jacobian
      TF_Tensor* grad_tensor = nullptr;
      TF_SessionRun(sess,
                    nullptr, // Run options.
                    &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                    &grad_op, &grad_tensor, 1, // Output tensors, output tensor values, number of outputs.
                    nullptr, 0, // Target operations, number of targets.
                    nullptr, // Run metadata.
                    status // Output status.
                    );
      const auto jacobian = static_cast<float*>(TF_TensorData(grad_tensor));

      cout<<" input ";
      for (int ido = 0; ido < n_feature; ido ++) {
          cout<<" "<<input_ref[ido];
      }
      cout<<endl;

      cout<<"output ";
      for (int ido = 0; ido < n_latent; ido ++) {
          cout<<" "<<output[ido];
      }
      cout<<endl;

      cout<<"jacobian ";
      for (int ido = 0; ido < n_latent; ido ++) {
          for (int idf = 0; idf < n_feature; idf ++) {
            cout<<" "<<jacobian[ido*n_feature+idf];
          }
      }
      cout<<endl;
      return 0;

}

