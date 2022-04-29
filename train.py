import torch


def train(model, train_loader, args, device, log_interval=1):
    datatype = torch.float64 if args.double else torch.float32
    N = train_loader.dataset.__len__()
    N_batch = train_loader.batch_size
    global_step = 0
    annealing_step = args.num_total_iter * args.warmup_portion

    if args.fixgp:
        model.raw_noise.requires_grad = False
        model.covar_module.raw_outputscale.requires_grad = False
        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.warmup_epochs), eta_min=1e-4)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        recon_sum = 0
        kl_sum = 0
        if epoch > args.warmup_epochs:
            scheduler.step()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, dtype=datatype)
            optimizer.zero_grad()
            output = model(data)
            if args.annealing:
                w = min((global_step/annealing_step), 1.0)
            else:
                w = 1.0
            loss = model.loss_function(data, N, N_batch, *output, klw=w)
            loss[0].backward()
            optimizer.step()
            global_step += 1
            with torch.no_grad():
                loss_sum += loss[0].item()
                recon_sum += loss[1].item()
                kl_sum += loss[2].item()

            if epoch % log_interval == 0 and batch_idx == len(train_loader)-1:
                # print(f"Epoch: {epoch}, Loss: {loss[0].item()}, Reconstruction: {loss[1].item()}, "
                #       f"kl_z2: {loss[2].item()}")
                print(f"Epoch: {epoch}, Loss: {loss_sum/len(train_loader)}, Reconstruction: {recon_sum/len(train_loader)}, "
                      f"kl_z2: {kl_sum/len(train_loader)}")
    torch.save(model.state_dict(), 'models/' + args.save + '.pth')

