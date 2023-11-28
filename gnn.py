import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import ComplEx

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## we are using complex's inbuilt loss for now. regularisation is part of weight decay of the optimiser
def lossfn(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
    ) -> torch.Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)

def train(model, optimizer, data_loader, batch_size=128, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = total_examples = 0
        for head_index, rel_type, tail_index in data_loader:
            optimizer.zero_grad()
            loss = model.loss(head_index, rel_type, tail_index)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, total_loss / total_examples))
    return 
        
        
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        return model.loss(data).item()
def test(model, data):
    model.eval()
    with torch.no_grad():
        return model.predict(data)
    
if __name__ == '__main__':
    import torch.optim as optim
    from torch_geometric.data import DataLoader
    from torch_geometric.nn.kge import ComplEx
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from generate_graph import graph_to_nx

    G = graph_to_nx("IMDB-Movie-Data.csv")

    data = torch_geometric.utils.from_networkx(G)

    embedding_dim = 64 ## undecided

    num_relations = torch.max(data.type).cpu().item() + 1

    model = ComplEx(data.num_nodes, num_relations, hidden_channels=embedding_dim).to(device)

    print(model)

    loader = model.loader(
        head_index= data.edge_index[0],
        rel_type= data.type,
        tail_index= data.edge_index[1],
        batch_size=128,
        shuffle=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-6)
    train(model,optimizer,loader)
    ### do something with the embeddings
    