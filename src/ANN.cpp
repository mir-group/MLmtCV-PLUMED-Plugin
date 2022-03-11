/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2018 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "tf_utils.hpp"
#include "Function.h"
#include "ActionRegister.h"

#include <cmath>

using namespace std;

namespace PLMD {
  namespace function {
  
    //+PLUMEDOC FUNCTION ANN
    /*
    Using an artificial neural netwwork to compute a set of collective variable from another set of other variables 
    
    The path where saved_models.pb sotres and the names of input, output and jacobian are provided as a string. Remember that the string should not be quoted by the quotation mark.
    
    Notice that ANN is not able to predict which will be periodic domain
    of the computed value automatically. 
    
    \par Examples
    
    The following input tells plumed to print the ANN output as computed from the square root of the square.
    \plumedfile
    DISTANCE LABEL=dist      ATOMS=3,5 COMPONENTS
    ANN  LABEL=distance2 ARG=dist.x,dist.y,dist.z MODELPATH=model INPUT=x OUTPUT=nn_return GRAD=grad_return
    PRINT ARG=dist,distance2
    \endplumedfile
    (See also \ref PRINT and \ref DISTANCE).
    
    */
    //+ENDPLUMEDOC
    
    
    class ANN : public Function
    {
      int n_feature;
      int n_latent;
      TF_Graph* graph;
      TF_Status* status;
      TF_Buffer* meta_graph_def;
      TF_Session* sess;
      TF_Output input_op, output_op, grad_op ;

      static int max_n_latent;

      public:
        explicit ANN(const ActionOptions&);
        ~ANN();
        void calculate();
        void load_session(const char*, const char*, const char*, const char*);
        static void registerKeywords(Keywords& keys);
    };

    int ANN::max_n_latent = 50;
    
    PLUMED_REGISTER_ACTION(ANN,"ANN")
    
    void ANN::registerKeywords(Keywords& keys) {

      Function::registerKeywords(keys);
    
      // need definition of the input argument 
      keys.use("ARG"); 

      // need module file file name
      keys.add("compulsory","MODELPATH", "the path of the saved_model.pb file");
      keys.add("compulsory","INPUT", "the tensor name of input in the model");
      keys.add("compulsory","OUTPUT", "the tensor name of output in the model");
      keys.add("compulsory","GRAD", "the tensor name of jacobian in the model");

      // reserve component place for the latent space
      componentsAreNotOptional(keys);
      for (int ido=0; ido < max_n_latent; ido++){
        string comp_name = to_string(ido);
        keys.addOutputComponent(comp_name,"default","the ith dimension of the output");
      }
    }

    void ANN::load_session(const char* export_dir, const char* input, const char* output, const char* grad){

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
      log.printf("ANN_TF: Loading export dir %s\n", export_dir);
      sess = TF_LoadSessionFromSavedModel(session_options,
                                          run_options,
                                          export_dir,
                                          tags,
                                          tags_len,
                                          graph,
                                          meta_graph_def,
                                          status);

      log.printf(TF_Message(status));

      if (graph == nullptr) {
        error("Can't load graph");
      } else {
        log.printf("ANN_TF: Tensorflow graph loaded\n");
      }

      // // print operations in the graph
      // TF_Operation* op;
      // std::size_t pos = 0;
      // while ((op = TF_GraphNextOperation(graph, &pos)) != nullptr) {
      //   const char* name = TF_OperationName(op);
      //   const int num_outputs = TF_OperationNumOutputs(op);
      //   const int num_inputs = TF_OperationNumInputs(op);
      //   log.printf("%d: %s input_size %d outputsize %d\n",  pos, name, num_inputs, num_outputs);
      // }

      input_op = {TF_GraphOperationByName(graph, input), 0};
      output_op = {TF_GraphOperationByName(graph, output), 0};
      grad_op = {TF_GraphOperationByName(graph, grad), 0};
      if (input_op.oper == nullptr) {
        char temp[100];
        sprintf(temp, "Can't find input %s in the graph", input);
        error(temp);
      }
      if (output_op.oper == nullptr) {
        char temp[100];
        sprintf(temp, "Can't find input %s in the graph", output);
        error(temp);
      }
      if (grad_op.oper == nullptr) {
        char temp[100];
        sprintf(temp, "Can't find input %s in the graph", grad);
        error(temp);
      }

      // read in input dimensions
      int num_dims = TF_GraphGetTensorNumDims(graph, input_op, status);
      vector<int64_t> dims(num_dims);
      TF_GraphGetTensorShape(graph, input_op, dims.data(), num_dims, status);
      n_feature = dims[1];

      num_dims = TF_GraphGetTensorNumDims(graph, output_op, status);
      TF_GraphGetTensorShape(graph, output_op, dims.data(), num_dims, status);
      n_latent = dims[1];

      // check whether input dimension is consistent with the argument dimension
      if (n_feature != int(getNumberOfArguments())) {
          char temp[100];
          sprintf(temp, "Number of arguments %d is not the same as the number of nodes in the model %d", getNumberOfArguments(), n_feature);
          error(temp);
      }

      if (n_latent > max_n_latent) {
          char temp[100];
          sprintf(temp, "Dimensions of latent space %d is larger than maximum %d, change the source code", n_latent, max_n_latent);
          error(temp);
      }

      log.printf("the input and output of the ANN is %d and %d \n", n_feature, n_latent);

    }
    
    ANN::ANN(const ActionOptions&ao):
      Action(ao),
      Function(ao)
    {

      //load the ann network
      string export_dir, input, output, grad;
      parse("MODELPATH", export_dir);
      parse("INPUT", input);
      parse("OUTPUT", output);
      parse("GRAD", grad);

      load_session(export_dir.c_str(), input.c_str(), output.c_str(), grad.c_str());
    
      checkRead();
     
      // register the latent space as collective variable component
      for (int ido=0; ido < n_latent; ido++){
        string comp_name = to_string(ido);
        addComponentWithDerivatives(comp_name);  
        componentIsNotPeriodic(comp_name); 
      }
      
    }
    
    void ANN::calculate() 
    {
      // vectorize the input CVs, note that there's a transform from double to float
      vector<float> input_vals(n_feature);
      for (int idx=0; idx<n_feature; idx++) {
        input_vals[idx] = getArgument(idx);
        // log.printf("original CVs %f", input_vals[idx]);
      }
      const vector<int64_t> input_dims = {1, n_feature};
      TF_Tensor* input_tensor = tf_utils::CreateTensor(
              TF_FLOAT, input_dims.data(), input_dims.size(), 
              input_vals.data(), input_vals.size() * sizeof(float));

//    if (TF_GetCode(status) != TF_OK) {
//      TF_DeleteStatus(status);
//      return 4;
//    }
//
      // forward calculation
      TF_Tensor* output_tensor = nullptr;
      TF_SessionRun(sess,
                    nullptr, // Run options.
                    &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                    &output_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                    nullptr, 0, // Target operations, number of targets.
                    nullptr, // Run metadata.
                    status // Output status.
                    );
      const auto output = static_cast<float*>(TF_TensorData(output_tensor));

      // // backbone test example
      // vector<double> output(n_latent);
      // output[0] = input_vals[0]+input_vals[1];
      // output[1] = input_vals[0]-input_vals[1];

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

      // // backbone test example
      // vector<vector<double>> jacobian;
      // jacobian.resize(n_latent, vector<double>(n_feature, 1.0));

      for (int ido = 0; ido < n_latent; ido ++) {

          // back_prop(derivatives_of_each_layer, ido);
          string comp_name = to_string(ido);
          Value* comp = getPntrToComponent(comp_name);
          // log.printf("output %d %f", ido, output[ido]);
          comp->set(output[ido]);
          for (int idf = 0; idf < n_feature; idf ++) {

            // TODO: test whether that is accumulative
            setDerivative(comp, idf, jacobian[ido*n_feature+idf]);  

          }

      #ifdef DEBUG_3
          printf("ANN_TF: derivatives = ");
          for (int idf = 0; idf < n_feature; idf ++) {
            printf("%f ", comp -> getDerivative(idf));
          }
          printf("\n");
      #endif
      }

      tf_utils::DeleteTensor(input_tensor);
      tf_utils::DeleteTensor(output_tensor);
      tf_utils::DeleteTensor(grad_tensor);
    
    }

    ANN::~ANN(){
      if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        error("Error run session");
      }
    
      TF_CloseSession(sess, status);
      if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        error("Error close session");
      }
    
      TF_DeleteSession(sess, status);
      if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        error("Error delete session");
      }
      tf_utils::DeleteGraph(graph);
      TF_DeleteStatus(status);
    }
    
  }
}

