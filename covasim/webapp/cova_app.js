const PlotlyChart = {
    props: ['graph'],
    render(h) {
        return h('div', {
            attrs: {
                id: this.graph.id,
            }
        })
    },

    mounted() {
        this.$nextTick(function () {
                let x = JSON.parse(this.graph.json)
                Plotly.react(this.graph.id, x.data, x.layout, {responsive: true});
            }
        )
    },
};


var vm = new Vue({
    el: '#app',

    components: {
        'plotly-chart': PlotlyChart,
    },

    data() {
        return {
            version: 'Version information unavailable',
            history: [],
            historyIdx: 0,
            sim_pars: {},
            epi_pars: {},
            result: { // Store currently displayed results
                graphs: [],
                summary: {},
                files: {},
            },
            running: false,
            err: '',
            reset_options: ['Example', 'Seattle'], // , 'Wuhan', 'Global'],
            reset_choice: 'Example'
        }
    },

    async created() {
        this.get_version();
        this.resetPars();
    },

    filters: {
        to2sf(value) {
            return Number(value).toFixed(2)
        }
    },

    methods: {

        async get_version() {
            let response = await sciris.rpc('get_version');
            this.version = response.data
        },

        async runSim() {
            this.running = true;
            // this.graphs = this.$options.data().graphs; // Uncomment this to clear the graphs on each run
            this.err = this.$options.data().err;

            console.log(this.status);
            console.log(this.sim_pars, this.epi_pars);

            // Run a a single sim
            try{
                let response = await sciris.rpc('run_sim', [this.sim_pars, this.epi_pars]);
                this.result.graphs = response.data.graphs;
                this.result.files = response.data.files;
                this.result.summary = response.data.summary;
                this.err = response.data.err;
                this.sim_pars= response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.history.push(JSON.parse(JSON.stringify({sim_pars:this.sim_pars, epi_pars:this.epi_pars, result:this.result})));
                this.historyIdx = this.history.length-1;

            } catch (e) {
                this.err = 'Error running model: ' + e;
            }
            this.running = false;

        },

        async resetPars() {
            let response = await sciris.rpc('get_defaults', [this.reset_choice]);
            this.sim_pars = response.data.sim_pars;
            this.epi_pars = response.data.epi_pars;
            this.graphs = [];
        },

        async downloadPars() {
            await sciris.download('download_pars', [this.sim_pars, this.epi_pars]);
        },

        async uploadPars() {
            try {
                let response = await sciris.upload('upload_pars');  //, [], {}, '');
                this.sim_pars = response.data.sim_pars;
                this.epi_pars = response.data.epi_pars;
                this.graphs = [];
            } catch (error) {
                sciris.fail(this, 'Could not upload parameters', error);
            }
        },

        loadPars(){
            this.sim_pars = this.history[this.historyIdx].sim_pars;
            this.epi_pars = this.history[this.historyIdx].epi_pars;
            this.result = this.history[this.historyIdx].result;
        },

        async downloadExcel() {
          let res = await fetch('data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + this.result.files.xlsx.content);
          let blob = await res.blob();
          saveAs(blob, this.result.files.xlsx.filename);
        },

        async downloadJson() {
          let res = await fetch('data:application/zip;base64,' + this.result.files.json.content);
          let blob = await res.blob();
          saveAs(blob, this.result.files.json.filename);
        },

      },

})