import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options_bdl.train_options import TrainOptions
import os
import numpy as np
from data_bdl import CreateSrcDataLoader
from data_bdl import CreateTrgDataLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
import tensorboardX

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def main():
    #load parameters######################################################################
    opt = TrainOptions()
    args = opt.initialize()

    #create timer##########################################################################
    _t = {'iter time' : Timer()}

    #create logs##########################################################################
    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)   
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    #Load data#############################################################################
    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    targetloader_iter, sourceloader_iter = iter(targetloader), iter(sourceloader)

    #Load deeplab+ optimizer###############################################################
    model, optimizer = CreateModel(args)

    #Load Descriminator+ optimizer#########################################################
    model_D, optimizer_D = CreateDiscriminator(args)
    



    #start training###############################################################################################
    start_iter = 0

    #restore checkoint##############################################################################
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])
        #start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[2])

    #Tensorboard####################################################################################    
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs", model_name))
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    model_D.train()
    model_D.cuda()
    loss = ['loss_seg_src', 'loss_seg_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real']
    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        
        model.adjust_learning_rate(args, optimizer, i)
        model_D.adjust_learning_rate(args, optimizer_D, i)
        
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        for param in model_D.parameters():
            param.requires_grad = False 
            
        src_img, src_lbl, _, _ = sourceloader_iter.next()
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()
        src_seg_score = model(src_img, lbl=src_lbl) # M(S),Ys
        loss_seg_src = model.loss   # Lseg(M(S),Ys)
        loss_seg_src.backward()
        


        if args.data_label_folder_target is not None:
            trg_img, trg_lbl, _, _ = targetloader_iter.next()
            trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda()
            trg_seg_score = model(trg_img, lbl=trg_lbl) 
            loss_seg_trg = model.loss
        else:
            trg_img, _, name = targetloader_iter.next()
            trg_img = Variable(trg_img).cuda()
            trg_seg_score = model(trg_img) # M(T)
            loss_seg_trg = 0
            

        		
        outD_trg = model_D(F.softmax(trg_seg_score), 0)
        loss_D_trg_fake = model_D.loss

        loss_trg = args.lambda_adv_target * loss_D_trg_fake + loss_seg_trg
        loss_trg.backward()
        
		
        for param in model_D.parameters():
            param.requires_grad = True
        
        src_seg_score, trg_seg_score = src_seg_score.detach(), trg_seg_score.detach()
        
        outD_src = model_D(F.softmax(src_seg_score), 0) # D(M(S), source)
        loss_D_src_real = model_D.loss / 2
        loss_D_src_real.backward()
		

        outD_trg = model_D(F.softmax(trg_seg_score), 1) # D(M(T), target)
        loss_D_trg_real = model_D.loss / 2
        loss_D_trg_real.backward()       
       
        
        optimizer.step()
        optimizer_D.step()
        
        
        for m in loss:
            train_writer.add_scalar(m, eval(m), i+1)
            
        if (i+1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source) +str(i+1)+'.pth' ))
            torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source) +str(i+1)+'_D.pth' ))   
            
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print('[it %d][src seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
            if i + 1 > args.num_steps_stop:
                print ('finish training')
                break
            _t['iter time'].tic()
            
if __name__ == '__main__':  
    main()
      
