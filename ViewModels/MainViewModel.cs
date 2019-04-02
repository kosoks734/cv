using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Microsoft.Win32;
using MVVM;

namespace People.ViewModels
{
    class MainViewModel : BaseViewModel
    {
        Models.ScriptRunner _model;
        string _videoInfo;
        string _loadPath;
        string _savePath;

        public MainViewModel() => this._model = new Models.ScriptRunner();

        public string VideoInfo
        {
            get => this._videoInfo;
            set
            {
                this._videoInfo = value;
                this.OnPropertyChanged(nameof(this.VideoInfo));
            }
        }

        public string LoadPath
        {
            get => this._loadPath;
            set
            {
                this._loadPath = value;
                this.OnPropertyChanged(nameof(this.LoadPath));
            }
        }

        public ICommand Load => new Command((obj) =>
        {
            var fd = new OpenFileDialog();
            this.LoadPath = "Suka";
            if (fd.ShowDialog() == true) this.LoadPath = fd.FileName;
        });
    }
}
