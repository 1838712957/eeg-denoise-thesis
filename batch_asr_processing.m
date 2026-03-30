%% DREAMS数据集批量ASR处理脚本 - 简化版
% 直接使用EEGLAB的pop函数处理EDF文件

clear; clc; close all;

%% 配置路径
project_root = 'C:\毕业论文';
input_dir = fullfile(project_root, '04_原始数据', 'Raw_edf 2');
output_dir = fullfile(project_root, '05_处理结果', 'ASR处理结果');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 启动EEGLAB
eeglab nogui

%% 获取EDF文件列表
edf_files = dir(fullfile(input_dir, 'subject*.edf'));
fprintf('发现 %d 个EDF文件\n', length(edf_files));

%% 批量处理
for i = 1:length(edf_files)
    filename = edf_files(i).name;
    subject_id = strrep(filename, '.edf', '');
    
    fprintf('\n========================================\n');
    fprintf('处理: %s (%d/%d)\n', subject_id, i, length(edf_files));
    fprintf('========================================\n');
    
    try
        % 1. 使用EEGLAB的biosig插件加载EDF文件
        filepath = fullfile(input_dir, filename);
        fprintf('  加载EDF文件...\n');
        EEG = pop_biosig(filepath);
        
        fprintf('    采样率: %d Hz\n', EEG.srate);
        fprintf('    通道数: %d\n', EEG.nbchan);
        
        % 2. 选择第一个通道
        if EEG.nbchan > 1
            fprintf('  选择第一个通道...\n');
            EEG = pop_select(EEG, 'channel', 1);
        end
        
        % 3. 重采样到128Hz（ASR要求采样率>110Hz）
        if EEG.srate ~= 128
            fprintf('  重采样到128Hz...\n');
            EEG = pop_resample(EEG, 128);
        end
        
        % 4. 带通滤波
        fprintf('  带通滤波 0.5-30Hz...\n');
        EEG = pop_eegfiltnew(EEG, 0.5, 30);
        
        % 5. ASR处理
        fprintf('  ASR处理...\n');
        if exist('pop_clean_rawdata', 'file') == 2
            EEG = pop_clean_rawdata(EEG, 'asr', 'on', 'asrcutoff', 20);
            fprintf('    ASR处理完成\n');
        else
            fprintf('    警告: clean_rawdata插件未安装，跳过ASR\n');
        end
        
        % 6. 保存结果
        output_filename = [subject_id '_ASR.set'];
        fprintf('  保存结果: %s\n', output_filename);
        pop_saveset(EEG, 'filename', output_filename, 'filepath', output_dir);
        
        fprintf('  完成!\n');
        
    catch ME
        fprintf('  错误: %s\n', ME.message);
        fprintf('  跳过此文件\n');
    end
end

fprintf('\n========================================\n');
fprintf('批量处理完成!\n');
fprintf('结果保存在: %s\n', output_dir);
fprintf('========================================\n');