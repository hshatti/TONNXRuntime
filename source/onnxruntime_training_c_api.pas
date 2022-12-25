unit onnxruntime_training_c_api;
{$IFDEF FPC}
  {$mode delphi}
  {$PACKRECORDS C}
  {$ModeSwitch advancedrecords}
  {$ModeSwitch typehelpers}
{$ENDIF}

{$MACRO ON}
{$define ORT_API_CALL:=stdcall}

{
Automatically converted by H2Pas 1.0.0 from onnxruntime_c_api_copy.h
The following command line parameters were used:
  onnxruntime_c_api_copy.h
}

{$H+}
interface

uses onnxruntime_pas_api;

type
  PPOrtTrainingSession = ^POrtTrainingSession;
  POrtTrainingSession = ^OrtTrainingSession;
  OrtTrainingSession = record end;  /// Type that enables performing training for the given user models.

  PPOrtCheckpointState = ^POrtCheckpointState;
  POrtCheckpointState = ^OrtCheckpointState;
  OrtCheckpointState = record end;  /// Type that holds the training states for the training session.

  OrtTrainingApi = record
    (** \brief Load a checkpoint state from directory on disk into checkpoint_state.
    *
    * This function will parse a checkpoint directory, pull relevant files and load the training
    * states into the checkpoint_state. This checkpoint state can then be used to create the
    * training session by invoking CreateTrainingSession. By doing so, the training session will resume
    * training from the given checkpoint.
    *
    * \param[in] checkpoint_path Path to the checkpoint directory
    * \param[out] checkpoint_state Checkpoint states that contains the states of the training session.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    LoadCheckpoint: function(const checkpoint_path: PORTCHAR_T;
                    checkpoint_state: PPOrtCheckpointState):POrtStatus; ORT_API_CALL;

    (** \brief Save the training session states to a checkpoint directory on disk.
    *
    * This function retrieves the training session states from the training session and serializes them
    * to a checkpoint directory on disk. This checkpoint can later be loaded by invoking LoadCheckpoint
    * to continue the training with the same states.
    *
    * \param[in] checkpoint_path Path to the checkpoint directory
    * \param[in] session The training session from where the checkpoint states are to be retrieved.
    * \param[in] save_optimizer_state Boolean flag indicating whether or not to save the optimizer states to the checkpoint.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    SaveCheckpoint: function(const checkpoint_path: PORTCHAR_T; const session: POrtTrainingSession;
                    save_optimizer_state: boolean):POrtStatus; ORT_API_CALL;

    (** \brief Create a training session that can be used to begin or resume training.
    *
    * This function creates a training session based on the env and session options provided that can
    * begin or resume training from a given checkpoint state for the given onnx models.
    * The checkpoint state represents the parameters of the training session which will be moved
    * to the device specified by the user through the session options (if necessary).
    *
    * \param[in] env Environment to be used for the training session.
    * \param[in] options Session options that the user can customize for this training session.
    * \param[in] checkpoint_state Training states that the training session uses as a starting point for training.
    * \param[in] train_model_path Model to be used to perform training that can be generated using the offline tooling library.
    * \param[in] eval_model_path Model to be used to perform evaluation that can be generated using the offline tooling library.
    * \param[in] optimizer_model_path Model to be used to the optimizer step for weight updates. The model can be generated using the offline tooling library.
    * \param[out] out Created training session.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    CreateTrainingSession: function(const env: POrtEnv; const options: POrtSessionOptions;
                    checkpoint_state: POrtCheckpointState; const train_model_path: PORTCHAR_T;
                    const eval_model_path: PORTCHAR_T; const optimizer_model_path: PORTCHAR_T;
                    _out: PPOrtTrainingSession):POrtStatus; ORT_API_CALL;

    (** \brief Retrieves the number of user outputs in the training model.
    *
    * This function returns the number of outputs of the training model so that the user can
    * allocate space for the number of outputs when TrainStep is invoked.
    *
    * \param[in] sess The training session which has working knowledge of the training model.
    * \param[out] out Number of user outputs in the training model.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    TrainingSessionGetTrainingModelOutputCount: function(const sess: POrtTrainingSession; _out: Psize_t ):POrtStatus; ORT_API_CALL;

    (** \brief Retrieves the number of user outputs in the eval model.
    *
    * This function returns the number of outputs of the eval model so that the user can
    * allocate space for the number of outputs when EvalStep is invoked.
    *
    * \param[in] sess The training session which has working knowledge of the eval model.
    * \param[out] out Number of user outputs in the eval model.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    TrainingSessionGetEvalModelOutputCount: function(const sess: POrtTrainingSession; _out: Psize_t):POrtStatus; ORT_API_CALL;

    TrainingSessionGetTrainingModelOutputName: function(const sess: POrtTrainingSession;index: size_t ; allocator: POrtAllocator; output: PPchar):POrtStatus; ORT_API_CALL;

    TrainingSessionGetEvalModelOutputName: function(const sess: POrtTrainingSession; index: size_t; allocator: POrtAllocator; output: PPchar):POrtStatus; ORT_API_CALL;

    (** \brief Reset the training model gradients to zero lazily.
    *
    * This function sets the internal state of the training session such that the training model gradients
    * will be reset just before the new gradients are computed on the next invocation of TrainStep.
    *
    * \param[in] session The training session which has working knowledge of the eval model.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    ResetGrad: function(session: POrtTrainingSession):POrtStatus; ORT_API_CALL;

    (** \brief Computes the outputs and the gradients for the training model for the given inputs
    *
    * This function performs a training step that computes the outputs and the gradients of the training model
    * for the given inputs. The train step is performed based on the training model that was provided
    * to the training session.
    * The gradients computed are stored inside the training session so they can be later consumed
    * by the OptimizerStep function.
    *
    * \param[in] sess The training session which has working knowledge of the eval model.
    * \param[in] run_options Run options for this training step.
    * \param[in] inputs_len Number of user inputs to the training model.
    * \param[in] inputs The user inputs to the training model.
    * \param[in] outputs_len Number of user outputs expected from this training step.
    * \param[out] outputs User outputs computed by train step.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    TrainStep: function( sess: POrtTrainingSession; const run_options: POrtRunOptions;
                    inputs_len: size_t; const inputs: PPOrtValue;
                    outputs_len: size_t; outputs: PPOrtValue):POrtStatus; ORT_API_CALL;

    (** \brief Computes the outputs for the eval model for the given inputs
    *
    * This function performs an eval step that computes the outputs of the eval model for the given inputs.
    * The eval step is performed based on the eval model that was provided to the training session.
    *
    * \param[in] sess The training session which has working knowledge of the eval model.
    * \param[in] run_options Run options for this eval step.
    * \param[in] inputs_len Number of user inputs to the eval model.
    * \param[in] inputs The user inputs to the eval model.
    * \param[in] outputs_len Number of user outputs expected from this eval step.
    * \param[out] outputs User outputs computed by eval step.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    EvalStep: function(const sess: POrtTrainingSession; const run_options: POrtRunOptions;
                    inputs_len: size_t; const inputs: PPOrtValue;
                    outputs_len: size_t; outputs: PPOrtValue):POrtStatus; ORT_API_CALL;

    (** \brief Sets the learning rate for this training session.
    *
    * This function allows users to set the learning rate for the training session. The current
    * learning rate is maintained by the training session and can be overwritten by invoking
    * this function with the desired learning rate. This function should not be used when a valid
    * learning rate scheduler is registered. It should be used either to set the learning rate
    * derived from a custom learning rate scheduler or to set the learning rate constant to be used
    * throughout the training session.
    * Please note that this function does not set the initial learning rate that may be needed
    * by the predefined learning rate schedulers. To set the initial learning rate for learning
    * rate schedulers, please look at the function `RegisterLinearLRScheduler`.
    *
    * \param[in] sess The training session on which the learning rate needs to be set.
    * \param[in] learning_rate Desired learning rate to set.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    SetLearningRate: function(sess: POrtTrainingSession; learning_rate: single):POrtStatus; ORT_API_CALL;

    (** \brief Gets the current learning rate for this training session.
    *
    * This function allows users to get the learning rate for the training session. The current
    * learning rate is maintained by the training session
    *
    * \param[in] sess The training session on which the learning rate needs to be set.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    GetLearningRate: function(sess: POrtTrainingSession; learning_rate: Psingle):POrtStatus; ORT_API_CALL;

    (** \brief Performs the weight updates for the trainable parameters using the optimizer model.
    *
    * This function performs the weight update step that updates the trainable parameters such that they
    * take a step in the direction of their gradients. The optimizer step is performed based on the optimizer
    * model that was provided to the training session.
    * The updated parameters are stored inside the training session so that they can be used by the next
    * TrainStep function call.
    *
    * \param[in] sess The training session which has working knowledge of the optimizer model.
    * \param[in] run_options Run options for this eval step.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    OptimizerStep: function(sess: POrtTrainingSession;
                    const run_options: POrtRunOptions):POrtStatus; ORT_API_CALL;

    (** \brief Registers the use of the Linear learning rate scheduler for the training session.
    *
    * Register a Linear learning rate scheduler with the given
    * learning rate scheduler parameters. Optionally specify the initial learning rate
    * that should be used with this learning rate scheduler and training session.
    *
    * \param[in] sess The training session that should use the linear learning rate scheduler.
    * \param[in] warmup_step_count Warmup steps for LR warmup.
    * \param[in] total_step_count Total step count.
    * \param[in] initial_lr The initial learning rate to be used by the training session.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    RegisterLinearLRScheduler: function(sess: POrtTrainingSession; const warmup_step_count: int64_t;
                      const total_step_count: int64_t; const initial_lr: single):POrtStatus; ORT_API_CALL;

    (** \brief Update the learning rate based on the registered learing rate scheduler.
    *
    * Takes a scheduler step that updates the learning rate that is being used by the training session.
    * This function should typically be called before invoking the optimizer step for each round,
    * or as determined necessary to update the learning rate being used by the training session.
    * Please note that a valid predefined learning rate scheduler must be first registered to invoke this
    * function.
    *
    * \param[in] sess The training session that has the registered learning rate scheduler.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    SchedulerStep: function(sess: POrtTrainingSession):POrtStatus; ORT_API_CALL;

    (** \brief Retrieves the size of all the parameters.
    *
    * Calculates the size of all the parameters for the training session.
    * When 'trainable_only' is true, the size is calculated for trainable params only.
    *
    * \param[in] sess The training session.
    * \param[in] trainable_only Whether to skip non-trainable parameters
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    GetParametersSize: function(sess: POrtTrainingSession;
                    _out: Psize_t; trainable_only: boolean):POrtStatus; ORT_API_CALL;

    (** \brief Copy parameters onto contiguous buffer held by parameters_buffer
    *
    * The parameters_buffer has to be of the size given by GetParametersSize api call,
    * with matching setting for 'trainable_only'. All the target parameters must be of the same
    * datatype. The OrtValue must be pre-allocated onto
    * the desired device. This is a complementary function to 'CopyBufferToParameters'.
    * Parameter ordering is preserved.
    * User is responsible for allocating/freeing the 'parameters_buffer'.
    *
    * \param[in] sess The training session.
    * \param[in] trainable_only Whether to skip non-trainable parameters
    * \param[out] parameters_buffer The pre-allocated OrtValue buffer to copy onto.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    CopyParametersToBuffer: function(sess: POrtTrainingSession;
                      parameters_buffer: POrtValue; trainable_only: boolean):POrtStatus; ORT_API_CALL;

    (** \brief Copy parameter values from contiguous buffer held by parameters_buffer onto parameters
    *
    * The parameters_buffer has to be of the size given by GetParametersSize api call,
    * with matching setting for 'trainable_only'. All the target parameters must be of the same
    * datatype. This is a complementary function to 'CopyBufferToParameters'
    * and can be used to load updated buffer values onto the parameters.
    * Parameter ordering is preserved.
    * User is responsible for allocating/freeing the 'parameters_buffer'.
    *
    * \param[in] sess The training session.
    * \param[in] trainable_only Whether to skip non-trainable parameters
    * \param[out] parameters_buffer The pre-allocated OrtValue buffer to copy from.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    CopyBufferToParameters: function(sess: POrtTrainingSession;
                      parameters_buffer: POrtValue; trainable_only: boolean):POrtStatus; ORT_API_CALL;

    (** \brief Frees up the memory used up by the training session.
    *
    * This function frees up any memory that was allocated in the training session. The training
    * session can no longer be used after this call.
    *
    *)
    ReleaseTrainingSession: procedure; ORT_API_CALL;

    (** \brief Frees up the memory used up by the checkpoint state.
    *
    * This function frees up any memory that was allocated in the checkpoint state. The checkpoint
    * state can no longer be used after this call.
    *
    *)
    CheckpointState: procedure; ORT_API_CALL;

    (** \brief Export a model that can be used for inferencing.
    *
    * If the training session was provided with an eval model, the training session can generate
    * an inference model if it knows the inference graph outputs. The input inference graph outputs
    * are used to prune the eval model so that the output model's outputs align with the provided outputs.
    * The exported model is saved at the path provided and can be used for inferencing with InferenceSession.
    * Note that the function re-loads the eval model from the path provided to CreateTrainingSession and expects
    * that this path still be valid.
    *
    * \param[in] sess The training session.
    * \param[in] inference_model_path Path where the inference model should be serialized to.
    *
    * \snippet{doc} snippets.dox OrtStatus Return Value
    *
    *)
    ExportModelForInferencing: function(sess: POrtTrainingSession;
                    const inference_model_path: PORTCHAR_T; graph_outputs_len: size_t;
                    const graph_output_names: PPchar):POrtStatus; ORT_API_CALL;

  end;


implementation

end.

