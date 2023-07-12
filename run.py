import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.utils import load_data, save_data
from model.MultiModal import MultiModal


def train(args, train_dataloader, dev_dataloader):
    model = MultiModal(args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for step, batch in enumerate(train_dataloader):
            ids, text, image, labels = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)
            labels = labels.to(device=args.device)

            optimizer.zero_grad()

            outputs = model(text=text, image=image)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            total_samples += labels.size(0)

        train_loss /= total_samples
        train_accuracy = train_correct / total_samples

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        evaluate(args, model, dev_dataloader, epoch)


def evaluate(args, model, dev_dataloader, epoch=None):
    model.eval()
    dev_loss = 0.0
    dev_correct = 0
    total_samples = 0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dev_dataloader:
            ids, text, image, labels = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)
            labels = labels.to(device=args.device)

            outputs = model(text=text, image=image)
            loss = loss_func(outputs, labels)

            dev_loss += loss.item() * labels.size(0)
            dev_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            total_samples += labels.size(0)

    dev_loss /= total_samples
    dev_accuracy = dev_correct / total_samples

    if epoch:
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f} (Epoch {epoch})")
    else:
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")


def test(args, test_dataloader, dev_dataloader):
    model = MultiModal(args).to(device=args.device)
    evaluate(args, model, dev_dataloader)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            ids, text, image, _ = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)

            outputs = model(text=text, image=image)
            predicted_labels = torch.argmax(outputs, dim=1)

            for i in range(len(ids)):
                item_id = ids[i]
                tag = test_dataloader.dataset.label_dict_str[int(predicted_labels[i])]
                prediction = {
                    'guid': item_id,
                    'tag': tag,
                }
                predictions.append(prediction)

    save_data(args.test_output_file, predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='训练')
    parser.add_argument('--do_test', action='store_true', help='测试')

    parser.add_argument('-test_output_file', '--test_output_file',
                        type=str, default='./test_with_label.txt', help='预测文件')

    parser.add_argument('-train_file', '--train_file',
                        type=str, default='./dataset/train.json', help='训练集')
    parser.add_argument('-dev_file', '--dev_file',
                        type=str, default='./dataset/dev.json', help='验证集')
    parser.add_argument('-test_file', '--test_file',
                        type=str, default='./dataset/test.json', help='测试集')
    parser.add_argument('-pretrained_model', '--pretrained_model',
                        type=str, default='roberta-base', help='预训练模型')

    parser.add_argument('-lr', '--lr',
                        type=float, default=1e-5, help='学习率')
    parser.add_argument('-dropout', '--dropout',
                        type=float, default=0.0, help='Dropout rate')
    parser.add_argument('-epochs', '--epochs',
                        type=int, default=10, help='迭代次数')
    parser.add_argument('-batch_size', '--batch_size',
                        type=int, default=4, help='Batch size')

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {args.device}')

    train_set, dev_set = load_data(args)
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_set, shuffle=False, batch_size=args.batch_size)

    if args.do_train:
        print('Model training...')
        train(args, train_dataloader, dev_dataloader)

    if args.do_test:
        test_set, dev_set = load_data(args)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size)
        print('Model testing...')
        test(args, test_dataloader, dev_dataloader)
