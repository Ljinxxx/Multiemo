import os
import torch


def build_checkpoint_config(trainer):
    return {
        'dataset': trainer.dataset,
        'batch_size': trainer.batch_size,
        'num_epochs': trainer.num_epochs,
        'learning_rate': trainer.learning_rate,
        'weight_decay': trainer.weight_decay,
        'num_layers': trainer.num_layers,
        'model_dim': trainer.model_dim,
        'num_heads': trainer.num_heads,
        'hidden_dim': trainer.hidden_dim,
        'dropout_rate': trainer.dropout_rate,
        'dropout_rec': trainer.dropout_rec,
        'temp_param': trainer.temp_param,
        'focus_param': trainer.focus_param,
        'sample_weight_param': trainer.sample_weight_param,
        'SWFC_loss_param': trainer.SWFC_loss_param,
        'HGR_loss_param': trainer.HGR_loss_param,
        'CE_loss_param': trainer.CE_loss_param,
        'aux_loss_param': trainer.aux_loss_param,
        'cmcl_loss_param': trainer.cmcl_loss_param,
        'cmcl_temp_param': trainer.cmcl_temp_param,
        'meld_label_smoothing': trainer.meld_label_smoothing,
        'use_graph_mtl': trainer.use_graph_mtl,
        'gnn_alpha': trainer.gnn_alpha,
        'gnn_edge_mode': trainer.gnn_edge_mode,
        'graph_emotion_loss_param': trainer.graph_emotion_loss_param,
        'identity_loss_param': trainer.identity_loss_param,
        'adv_identity_loss_param': trainer.adv_identity_loss_param,
        'ortho_loss_param': trainer.ortho_loss_param,
        'grl_lambda': trainer.grl_lambda,
        'multi_attn_flag': trainer.multi_attn_flag
    }


def save_disentangle_checkpoint(trainer, checkpoint_path):
    os.makedirs(
        os.path.dirname(checkpoint_path),
        exist_ok=True
    )

    checkpoint = {
        'dataset': trainer.dataset,
        'best_test_f1': trainer.best_test_f1,
        'best_epoch': trainer.best_epoch,
        'model_state_dict': trainer.model.state_dict(),
        'graph_mtl_state_dict': trainer.graph_mtl.state_dict(),
        'config': build_checkpoint_config(trainer)
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    return checkpoint_path
