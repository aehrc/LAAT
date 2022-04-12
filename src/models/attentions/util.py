from src.models.attentions.attention_layer import *


def init_attention_layer(model):
    if model.args.joint_mode == "flat":
        if model.attention_mode is not None:
            model.attention = AttentionLayer(args=model.args, size=model.output_size,
                                             n_labels=model.vocab.all_n_labels(), n_level=model.vocab.n_level())

        model.linears = nn.ModuleList([nn.Linear(model.output_size + model.vocab.n_labels(level))
                                       for level in range(model.vocab.n_level())])

    elif model.args.joint_mode == "hierarchical":
        model.level_projection_size = model.args.level_projection_size
        if model.attention_mode is not None:
            model.attention = AttentionLayer(args=model.args, size=model.output_size,
                                             level_projection_size=model.level_projection_size,
                                             n_labels=model.vocab.all_n_labels(), n_level=model.vocab.n_level())
        linears = []
        projection_linears = []
        for level in range(model.vocab.n_level()):
            level_projection_size = 0 if level == 0 else model.level_projection_size
            linears.append(nn.Linear(model.output_size + level_projection_size,
                                     model.vocab.n_labels(level)))
            projection_linears.append(nn.Linear(model.vocab.n_labels(level), model.level_projection_size, bias=False))
        model.linears = nn.ModuleList(linears)
        model.projection_linears = nn.ModuleList(projection_linears)
    elif model.args.joint_mode == "hicu":
        model.attention = Decoder(args=model.args, input_size=model.output_size, Y=model.vocab.all_n_labels(), dicts=model.vocab)
    else:
        raise NotImplementedError
    if model.attention_mode is not None:
        model.r = model.attention.r


def perform_attention(model, all_output, last_output):
    attention_weights = None
    if model.args.joint_mode == "flat":
        if model.attention_mode is not None:
            attention_outputs = [model.attention(all_output, label_level=label_lvl)
                                 for label_lvl in range(model.vocab.n_level())]
            weighted_outputs = [attention_outputs[label_lvl][0] for label_lvl in range(model.vocab.n_level())]
            attention_weights = [attention_outputs[label_lvl][1] for label_lvl in range(model.vocab.n_level())]

            if model.attention_mode not in ["label", "caml"]:
                if model.use_dropout:
                    for label_lvl in range(model.vocab.n_level()):
                        weighted_outputs[label_lvl] = model.dropout(weighted_outputs[label_lvl])
                for label_lvl in range(model.vocab.n_level()):
                    weighted_outputs[label_lvl] = model.linears[label_lvl](weighted_outputs[label_lvl])
        else:

            weighted_outputs = []
            if model.use_dropout:
                for label_lvl in range(model.vocab.n_level()):
                    weighted_outputs.append(model.dropout(last_output))
            else:
                weighted_outputs = [last_output] * model.vocab.n_level()

            for label_lvl in range(model.vocab.n_level()):
                weighted_outputs[label_lvl] = model.linears[label_lvl](weighted_outputs[label_lvl])

    elif model.args.joint_mode == "hierarchical":
        previous_level_projection = None
        if model.attention_mode is not None:
            weighted_outputs = []
            attention_weights = []
            for level in range(model.vocab.n_level()):
                weighted_output, attention_weight = model.attention(all_output,
                                                                    previous_level_projection, label_level=level)
                if model.attention_mode not in ["label", "caml"]:
                    if model.use_dropout:
                        weighted_output = model.dropout(weighted_output)
                    weighted_output = model.linears[level](weighted_output)

                previous_level_projection = model.projection_linears[level](
                    torch.sigmoid(weighted_output) if model.attention_mode in ["label", "caml"]
                    else torch.softmax(weighted_output, 1))
                previous_level_projection = F.sigmoid(previous_level_projection)
                weighted_outputs.append(weighted_output)
                attention_weights.append(attention_weight)
        else:
            weighted_outputs = []
            attention_weights = None
            previous_level_projection = None
            for level in range(model.vocab.n_level()):
                if previous_level_projection is not None:
                    last_output = [last_output, previous_level_projection]
                    last_output = torch.cat(last_output, dim=1)

                output = last_output
                if model.use_dropout:
                    output = model.dropout(last_output)

                output = model.linears[level](output)
                weighted_outputs.append(output)
                previous_level_projection = model.projection_linears[level](torch.softmax(output, 1))
    elif model.args.joint_mode == "hicu":
        weighted_output, attention_weight = model.attention(all_output)
        weighted_outputs = [weighted_output]
        attention_weights = [attention_weight]
    return weighted_outputs, attention_weights
